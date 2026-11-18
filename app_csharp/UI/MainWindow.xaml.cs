using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Win32;
using SoundPulse.AI;
using SoundPulse.Audio;
using SoundPulse.Domain;
using SoundPulse.Services;
using SoundPulse.Storage;

namespace SoundPulse.UI;

public partial class MainWindow : Window
{
    private readonly string[] _startupArgs;
    private readonly AudioPlayerEngine _player = new();
    private readonly List<TrackItem> _allTracks = [];
    private readonly HashSet<string> _trackPathIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly ObservableCollection<TrackItem> _visibleTracks = [];

    private readonly ObservableCollection<CategoryFilterItem> _libraryFilters = [];
    private readonly ObservableCollection<CategoryFilterItem> _vibeFilters = [];
    private readonly ObservableCollection<CategoryFilterItem> _genreFilters = [];
    private readonly ObservableCollection<SearchResultItem> _searchResults = [];

    private readonly DispatcherTimer _playbackTimer;
    private readonly float[] _meterLevels = new float[12];
    private readonly Border[] _visualizerBars = new Border[12];
    private readonly Random _rng = new();
    private readonly MusicDownloader _downloader = new();
    private readonly WatchFolderService _watchFolder;

    private AppSettings _settings = new();
    private MusicClassifier? _classifier;
    private AnalysisWorker? _analysisWorker;
    private CategoryFilterItem _activeFilter = null!;
    private TrackItem? _currentTrack;
    private TrackItem? _inspectedTrack;

    private bool _isUserSeeking;
    private bool _isRepeat;
    private bool _isShuffle;
    private bool _isClosing;
    private bool _isSyncingSettings;
    private int _queuedBatchTotal;
    private int _queuedBatchCompleted;
    private CancellationTokenSource? _coverCts;
    private CancellationTokenSource? _searchCts;

    public MainWindow(string[] startupArgs)
    {
        _startupArgs = startupArgs;
        InitializeComponent();

        BuildVisualizerBars();
        InitializeDefaultFilters();

        TrackListBox.ItemsSource = _visibleTracks;
        LibraryFiltersList.ItemsSource = _libraryFilters;
        VibeFiltersList.ItemsSource = _vibeFilters;
        GenreFiltersList.ItemsSource = _genreFilters;
        SearchResultsList.ItemsSource = _searchResults;

        // Register with handledEventsToo=true so Slider's internal class handlers never swallow seek clicks!
        SeekSlider.AddHandler(PreviewMouseLeftButtonDownEvent, new MouseButtonEventHandler(SeekSlider_PreviewMouseDown), handledEventsToo: true);
        SeekSlider.AddHandler(PreviewMouseLeftButtonUpEvent, new MouseButtonEventHandler(SeekSlider_PreviewMouseUp), handledEventsToo: true);

        _watchFolder = new WatchFolderService(OnWatchFolderAudioArrived);

        // Timer ONLY runs while a track is actively playing (0% CPU when idle/paused!)
        _playbackTimer = new DispatcherTimer(DispatcherPriority.Render)
        {
            Interval = TimeSpan.FromMilliseconds(50)
        };
        _playbackTimer.Tick += PlaybackTimer_Tick;

        _player.TrackEnded += () => Dispatcher.BeginInvoke(OnTrackEnded);
        _player.PlaybackStateChanged += () => Dispatcher.BeginInvoke(SyncPlaybackUiState);

        Loaded += MainWindow_Loaded;
        Closing += MainWindow_Closing;
    }

    private void BuildVisualizerBars()
    {
        VisualizerGrid.Children.Clear();
        for (int i = 0; i < _visualizerBars.Length; i++)
        {
            var bar = new Border
            {
                Width = 6,
                Height = 3,
                CornerRadius = new CornerRadius(3),
                VerticalAlignment = VerticalAlignment.Bottom,
                HorizontalAlignment = HorizontalAlignment.Center,
                Background = new SolidColorBrush(Color.FromArgb(190, 129, 140, 248))
            };
            _visualizerBars[i] = bar;
            VisualizerGrid.Children.Add(bar);
        }
    }

    private void InitializeDefaultFilters()
    {
        _activeFilter = new CategoryFilterItem
        {
            Key = "all",
            Kind = FilterKind.All,
            Label = "Все треки",
            Icon = "♫",
            AccentHex = "#818CF8",
            IsSelected = true
        };
        _libraryFilters.Add(_activeFilter);

        string[] defaultVibes =
        [
            "Драйв / Энергия",
            "Вечеринка / Танцы",
            "Чилл / Отдых",
            "Лирика / Грусть",
            "Фокус / Концентрация"
        ];
        for (int i = 0; i < defaultVibes.Length; i++)
        {
            _vibeFilters.Add(new CategoryFilterItem
            {
                Key = $"vibe:{i}",
                Kind = FilterKind.Vibe,
                VibeIndex = i,
                Label = defaultVibes[i],
                Icon = CategoryPalette.GetVibeIcon(i),
                AccentHex = CategoryPalette.GetVibeColor(i, defaultVibes[i])
            });
        }

        string[] defaultGenres =
        [
            "blues", "classical", "country", "electronic", "folk", "hiphop",
            "jazz", "lofi_ambient", "metal", "pop", "reggae", "rock"
        ];
        foreach (string g in defaultGenres)
        {
            _genreFilters.Add(new CategoryFilterItem
            {
                Key = g,
                Kind = FilterKind.Genre,
                Label = CategoryPalette.FormatGenreName(g),
                Icon = "•",
                AccentHex = CategoryPalette.GetGenreColor(g)
            });
        }
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        // Load settings & configure Watch Folder
        _settings = AppSettings.Load();
        ApplySettingsToUi();

        // Load embedded ONNX model off the UI thread for instant window paint
        try
        {
            var classifier = await Task.Run(() => new MusicClassifier());
            if (_isClosing)
            {
                classifier.Dispose();
                return;
            }

            _classifier = classifier;
            _analysisWorker = new AnalysisWorker(_classifier, OnTrackAnalysisUpdated);

            SyncLabelsFromModel(_classifier.Labels);

            ModelStatusDot.Fill = CreateFrozenBrush("#10B981");
            ModelStatusText.Text = $"{_classifier.Labels.ModelName} · ONNX готов";
        }
        catch (Exception ex)
        {
            ModelStatusDot.Fill = CreateFrozenBrush("#F43F5E");
            ModelStatusText.Text = $"AI недоступен: {ex.Message}";
        }

        // Restore cached library off the UI thread
        var cached = await Task.Run(LibraryCache.Load);
        if (_isClosing) return;

        RestoreCachedLibrary(cached);

        // Auto-pick up any audio files in SoundPulse's own DownloadFolder (e.g. previously downloaded tracks)
        if (Directory.Exists(_settings.DownloadFolder))
        {
            await AddPathsAsync([_settings.DownloadFolder]);
        }

        // If files/folders were passed via command line (e.g. dragged onto SoundPulse.exe)
        if (_startupArgs.Length > 0)
        {
            await AddPathsAsync(_startupArgs);
        }
    }

    private void SyncLabelsFromModel(ModelLabels labels)
    {
        if (labels.Vibes.Count > 0)
        {
            _vibeFilters.Clear();
            for (int i = 0; i < labels.Vibes.Count; i++)
            {
                string vibeName = labels.Vibes[i];
                _vibeFilters.Add(new CategoryFilterItem
                {
                    Key = $"vibe:{i}",
                    Kind = FilterKind.Vibe,
                    VibeIndex = i,
                    Label = vibeName,
                    Icon = CategoryPalette.GetVibeIcon(i),
                    AccentHex = CategoryPalette.GetVibeColor(i, vibeName)
                });
            }
            InspectorVibesHeader.Text = $"НАСТРОЕНИЕ ({labels.Vibes.Count} {(labels.Vibes.Count == 5 ? "КЛАССОВ" : "КЛАССА")})";
        }

        if (labels.Genres.Count > 0)
        {
            _genreFilters.Clear();
            foreach (string g in labels.Genres)
            {
                _genreFilters.Add(new CategoryFilterItem
                {
                    Key = g,
                    Kind = FilterKind.Genre,
                    Label = CategoryPalette.FormatGenreName(g),
                    Icon = "•",
                    AccentHex = CategoryPalette.GetGenreColor(g)
                });
            }
            GenresHeaderText.Text = $"ЖАНРЫ ({labels.Genres.Count} КЛАССОВ)";
        }
    }

    private int CurrentCacheVersion => LibraryCache.ComputeCacheVersion(_classifier?.Labels);

    private void RestoreCachedLibrary(List<CachedTrackRecord> records)
    {
        var needsReanalysis = new List<TrackItem>();
        int expectedVersion = CurrentCacheVersion;

        foreach (var rec in records)
        {
            if (string.IsNullOrWhiteSpace(rec.FilePath) || !File.Exists(rec.FilePath))
                continue;
            if (!_trackPathIndex.Add(rec.FilePath))
                continue;

            var fi = new FileInfo(rec.FilePath);
            var item = new TrackItem
            {
                FilePath = rec.FilePath,
                FileSize = fi.Length,
                LastWriteUtcTicks = fi.LastWriteTimeUtc.Ticks,
                Title = string.IsNullOrWhiteSpace(rec.Title) ? Path.GetFileNameWithoutExtension(rec.FilePath) : rec.Title,
                Artist = string.IsNullOrWhiteSpace(rec.Artist) ? "Неизвестный исполнитель" : rec.Artist,
                Album = rec.Album,
                DurationSec = rec.DurationSec
            };

            bool cacheValid =
                rec.AnalysisVersion == expectedVersion &&
                rec.FileSize == fi.Length &&
                rec.LastWriteUtcTicks == fi.LastWriteTimeUtc.Ticks &&
                !string.IsNullOrWhiteSpace(rec.Genre) &&
                rec.VibeIndex >= 0;

            if (cacheValid)
            {
                item.Genre = rec.Genre;
                item.GenreConfidence = rec.GenreConfidence;
                item.Vibe = rec.Vibe;
                item.VibeIndex = rec.VibeIndex;
                item.VibeConfidence = rec.VibeConfidence;
                item.Bpm = rec.Bpm;
                item.Energy = rec.Energy;
                item.TopGenres = rec.TopGenres;
                item.VibeScores = rec.VibeScores;
                item.Status = TrackStatus.Ready;
            }
            else
            {
                item.Status = TrackStatus.Pending;
                needsReanalysis.Add(item);
            }

            _allTracks.Add(item);
        }

        RefreshCountsAndFilter();

        if (needsReanalysis.Count > 0 && _analysisWorker != null)
        {
            EnqueueForAnalysis(needsReanalysis);
        }
    }

    private async Task<List<TrackItem>> AddPathsAsync(IEnumerable<string> inputPaths, bool markForAutoSort = false)
    {
        var discoveredFiles = await Task.Run(() =>
        {
            var files = new List<string>();
            foreach (string raw in inputPaths)
            {
                if (File.Exists(raw) && AudioDecoder.IsSupportedAudioFile(raw))
                {
                    files.Add(raw);
                }
                else if (Directory.Exists(raw))
                {
                    try
                    {
                        foreach (string f in Directory.EnumerateFiles(raw, "*.*", SearchOption.AllDirectories))
                        {
                            if (AudioDecoder.IsSupportedAudioFile(f))
                                files.Add(f);
                        }
                    }
                    catch
                    {
                        // Skip inaccessible subdirectories
                    }
                }
            }
            return files;
        });

        var distinctFiles = discoveredFiles.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        var existingTracks = _allTracks
            .Where(t => distinctFiles.Contains(t.FilePath, StringComparer.OrdinalIgnoreCase))
            .ToList();

        var newFiles = distinctFiles.Where(f => !_trackPathIndex.Contains(f)).ToList();
        if (newFiles.Count == 0)
            return existingTracks;

        // Read tags off UI thread
        var createdTracks = await Task.Run(() =>
        {
            var items = new List<TrackItem>(newFiles.Count);
            foreach (string path in newFiles)
            {
                var fi = new FileInfo(path);
                var meta = AudioDecoder.ReadMetadata(path);
                items.Add(new TrackItem
                {
                    FilePath = path,
                    FileSize = fi.Length,
                    LastWriteUtcTicks = fi.LastWriteTimeUtc.Ticks,
                    Title = meta.Title,
                    Artist = meta.Artist,
                    Album = meta.Album,
                    DurationSec = meta.DurationSec,
                    AutoSortPending = markForAutoSort,
                    Status = TrackStatus.Pending
                });
            }
            return items;
        });

        foreach (var item in createdTracks)
        {
            if (_trackPathIndex.Add(item.FilePath))
                _allTracks.Add(item);
        }

        RefreshCountsAndFilter();
        EnqueueForAnalysis(createdTracks);

        if (_inspectedTrack == null && createdTracks.Count > 0)
        {
            ShowTrackInInspector(createdTracks[0]);
        }

        existingTracks.AddRange(createdTracks);
        return existingTracks;
    }

    private void FocusTracksInLibrary(IReadOnlyList<TrackItem> tracks)
    {
        if (tracks.Count == 0) return;

        ShowLibraryView();

        // Switch to "All Tracks" so the newly downloaded track is guaranteed visible
        foreach (var f in _libraryFilters) f.IsSelected = false;
        foreach (var f in _vibeFilters) f.IsSelected = false;
        foreach (var f in _genreFilters) f.IsSelected = false;

        _libraryFilters[0].IsSelected = true;
        _activeFilter = _libraryFilters[0];
        if (!string.IsNullOrEmpty(SearchBox.Text))
            SearchBox.Text = string.Empty;

        RefreshCountsAndFilter();

        var target = tracks[^1];
        TrackListBox.SelectedItem = target;
        TrackListBox.ScrollIntoView(target);
        ShowTrackInInspector(target);
    }

    private void EnqueueForAnalysis(IReadOnlyList<TrackItem> tracks)
    {
        if (_analysisWorker == null || tracks.Count == 0) return;

        _queuedBatchTotal += tracks.Count;
        UpdateAnalysisBanner();

        foreach (var t in tracks)
        {
            t.Status = TrackStatus.Pending;
            _analysisWorker.Enqueue(t);
        }
    }

    private void OnTrackAnalysisUpdated(TrackItem track, TrackStatus status, AnalysisResult? result, string? error)
    {
        Dispatcher.BeginInvoke(() =>
        {
            if (_isClosing) return;

            if (status == TrackStatus.Analyzing)
            {
                track.Status = TrackStatus.Analyzing;
                return;
            }

            if (status == TrackStatus.Ready && result != null)
            {
                track.ApplyAnalysis(result);

                // Optional auto-sort into <DownloadFolder>\<Vibe>\<Genre>\<FileName>
                if (track.AutoSortPending && _settings.AutoSortByCategory && _currentTrack != track)
                {
                    track.AutoSortPending = false;
                    TryAutoSortTrackFile(track);
                }
            }
            else if (status == TrackStatus.Error)
            {
                track.ErrorMessage = error ?? "Не удалось декодировать аудиофайл";
                track.Status = TrackStatus.Error;
            }

            _queuedBatchCompleted++;
            UpdateAnalysisBanner();
            RefreshCountsAndFilter();

            if (_inspectedTrack == track)
            {
                ShowTrackInInspector(track);
            }
            if (_currentTrack == track)
            {
                UpdateNowPlayingBar();
            }

            if (_queuedBatchCompleted >= _queuedBatchTotal)
            {
                _queuedBatchTotal = 0;
                _queuedBatchCompleted = 0;
                UpdateAnalysisBanner();
                var snapshot = _allTracks.ToList();
                int ver = CurrentCacheVersion;
                _ = Task.Run(() => LibraryCache.Save(snapshot, ver));
            }
        });
    }

    private void UpdateAnalysisBanner()
    {
        if (_queuedBatchTotal > 0 && _queuedBatchCompleted < _queuedBatchTotal)
        {
            AnalysisBanner.Visibility = Visibility.Visible;
            AnalysisProgressText.Text = $"AI-анализ треков: {_queuedBatchCompleted} из {_queuedBatchTotal}…";
            AnalysisProgressBar.Value = 100.0 * _queuedBatchCompleted / _queuedBatchTotal;
        }
        else
        {
            AnalysisBanner.Visibility = Visibility.Collapsed;
        }
    }

    private void RefreshCountsAndFilter()
    {
        // Update sidebar counters
        _libraryFilters[0].Count = _allTracks.Count;

        foreach (var vf in _vibeFilters)
            vf.Count = _allTracks.Count(t => t.Status == TrackStatus.Ready && t.VibeIndex == vf.VibeIndex);

        foreach (var gf in _genreFilters)
            gf.Count = _allTracks.Count(t => t.Status == TrackStatus.Ready &&
                t.Genre.Equals(gf.Key, StringComparison.OrdinalIgnoreCase));

        // Rebuild filtered track list
        string query = SearchBox.Text.Trim();
        var filtered = _allTracks.Where(t => MatchesFilter(t, _activeFilter, query)).ToList();

        var prevSelected = TrackListBox.SelectedItem as TrackItem;
        _visibleTracks.Clear();
        foreach (var t in filtered)
            _visibleTracks.Add(t);

        if (prevSelected != null && _visibleTracks.Contains(prevSelected))
        {
            TrackListBox.SelectedItem = prevSelected;
        }

        // Update header & empty state
        ActiveCategoryTitle.Text = _activeFilter.Label;
        ActiveCategoryDot.Fill = CreateFrozenBrush(_activeFilter.AccentHex);

        int count = _visibleTracks.Count;
        string noun = PluralizeTracks(count);
        ActiveCategorySubtitle.Text = _allTracks.Count == 0
            ? "Добавьте аудиофайлы или перетащите папку в окно"
            : $"{count} {noun}" + (string.IsNullOrEmpty(query) ? "" : $" по запросу «{query}»");

        bool showEmpty = _visibleTracks.Count == 0;
        EmptyStatePanel.Visibility = showEmpty ? Visibility.Visible : Visibility.Collapsed;
        if (showEmpty)
        {
            if (_allTracks.Count == 0)
            {
                EmptyStateTitle.Text = "Перетащите музыку сюда";
                EmptyStateDesc.Text = "Поддерживаются MP3, WAV, FLAC, OGG, M4A, AAC, WMA. Модель MusicAudioNet автоматически определит жанр и вайб каждого трека.";
            }
            else
            {
                EmptyStateTitle.Text = "В этой категории пока пусто";
                EmptyStateDesc.Text = "Попробуйте выбрать другую категорию, сбросить поиск или добавить новые треки.";
            }
        }
    }

    private static bool MatchesFilter(TrackItem t, CategoryFilterItem filter, string query)
    {
        bool categoryMatch = filter.Kind switch
        {
            FilterKind.All => true,
            FilterKind.Vibe => t.Status == TrackStatus.Ready && t.VibeIndex == filter.VibeIndex,
            FilterKind.Genre => t.Status == TrackStatus.Ready && t.Genre.Equals(filter.Key, StringComparison.OrdinalIgnoreCase),
            _ => true
        };

        if (!categoryMatch) return false;
        if (string.IsNullOrEmpty(query)) return true;

        return t.Title.Contains(query, StringComparison.OrdinalIgnoreCase)
            || t.Artist.Contains(query, StringComparison.OrdinalIgnoreCase)
            || t.Album.Contains(query, StringComparison.OrdinalIgnoreCase)
            || t.GenreDisplay.Contains(query, StringComparison.OrdinalIgnoreCase)
            || t.Vibe.Contains(query, StringComparison.OrdinalIgnoreCase);
    }

    private void PlayTrack(TrackItem track)
    {
        if (!File.Exists(track.FilePath))
        {
            MessageBox.Show($"Файл не найден:\n{track.FilePath}", "SoundPulse AI", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            if (_currentTrack != null)
                _currentTrack.IsPlaying = false;

            _currentTrack = track;
            _currentTrack.IsPlaying = true;

            _player.LoadAndPlay(track.FilePath);

            ShowTrackInInspector(track);
            UpdateNowPlayingBar();
            SyncPlaybackUiState();
        }
        catch (Exception ex)
        {
            track.IsPlaying = false;
            MessageBox.Show($"Не удалось воспроизвести трек:\n{ex.Message}", "SoundPulse AI", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void OnTrackEnded()
    {
        if (_isClosing || _currentTrack == null) return;

        if (_isRepeat)
        {
            PlayTrack(_currentTrack);
            return;
        }

        PlayNextTrack();
    }

    private void PlayNextTrack()
    {
        var list = _visibleTracks.Count > 0 ? _visibleTracks.ToList() : _allTracks;
        if (list.Count == 0) return;

        if (_isShuffle && list.Count > 1)
        {
            TrackItem pick;
            do { pick = list[_rng.Next(list.Count)]; } while (pick == _currentTrack);
            PlayTrack(pick);
            return;
        }

        int idx = _currentTrack != null ? list.IndexOf(_currentTrack) : -1;
        int nextIdx = (idx + 1) % list.Count;
        PlayTrack(list[nextIdx]);
    }

    private void PlayPreviousTrack()
    {
        // If more than 3 seconds into track, restart current track first
        if (_player.HasTrack && _player.Position.TotalSeconds > 3.0)
        {
            _player.Seek(TimeSpan.Zero);
            UpdateSeekUi();
            return;
        }

        var list = _visibleTracks.Count > 0 ? _visibleTracks.ToList() : _allTracks;
        if (list.Count == 0) return;

        int idx = _currentTrack != null ? list.IndexOf(_currentTrack) : 0;
        int prevIdx = idx <= 0 ? list.Count - 1 : idx - 1;
        PlayTrack(list[prevIdx]);
    }

    private void SyncPlaybackUiState()
    {
        bool playing = _player.IsPlaying;
        PlayIconPath.Visibility = playing ? Visibility.Collapsed : Visibility.Visible;
        PauseIconPath.Visibility = playing ? Visibility.Visible : Visibility.Collapsed;

        if (_currentTrack != null)
            _currentTrack.IsPlaying = playing;

        if (playing)
        {
            if (!_playbackTimer.IsEnabled)
                _playbackTimer.Start();
        }
        else
        {
            // Stop timer immediately when paused/stopped -> 0% idle CPU!
            if (_playbackTimer.IsEnabled)
                _playbackTimer.Stop();

            ResetVisualizerBars();
        }

        UpdateSeekUi();
    }

    private void PlaybackTimer_Tick(object? sender, EventArgs e)
    {
        if (!_player.IsPlaying)
        {
            _playbackTimer.Stop();
            ResetVisualizerBars();
            return;
        }

        UpdateSeekUi();

        // Update 12-band FFT visualizer ONLY when inspecting the currently playing track
        if (_inspectedTrack == _currentTrack)
        {
            _player.CopyMeterLevels(_meterLevels);
            var accent = _currentTrack != null
                ? CreateFrozenBrush(_currentTrack.VibeColorHex)
                : CreateFrozenBrush("#818CF8");

            for (int i = 0; i < _visualizerBars.Length; i++)
            {
                double h = 3.0 + _meterLevels[i] * 39.0;
                _visualizerBars[i].Height = h;
                _visualizerBars[i].Background = accent;
            }
        }
        else
        {
            ResetVisualizerBars();
        }
    }

    private void ResetVisualizerBars()
    {
        for (int i = 0; i < _visualizerBars.Length; i++)
        {
            _visualizerBars[i].Height = 3.0;
        }
    }

    private void UpdateSeekUi()
    {
        if (!_player.HasTrack)
        {
            TimeCurrentText.Text = "00:00";
            TimeTotalText.Text = "00:00";
            if (!_isUserSeeking) SeekSlider.Value = 0;
            return;
        }

        var pos = _player.Position;
        var dur = _player.Duration;
        TimeCurrentText.Text = FormatTime(pos);
        TimeTotalText.Text = FormatTime(dur);

        if (!_isUserSeeking && dur.TotalSeconds > 0)
        {
            SeekSlider.Maximum = dur.TotalSeconds;
            SeekSlider.Value = Math.Clamp(pos.TotalSeconds, 0, dur.TotalSeconds);
        }
    }

    private void UpdateNowPlayingBar()
    {
        if (_currentTrack == null)
        {
            NowPlayingTitle.Text = "Нет активного трека";
            NowPlayingSubtitle.Text = "Двойной клик по треку для запуска";
            MiniCoverMonogram.Text = "♪";
            MiniCoverImage.Visibility = Visibility.Collapsed;
            return;
        }

        NowPlayingTitle.Text = _currentTrack.Title;
        string suffix = _currentTrack.Status == TrackStatus.Ready
            ? $"  ·  {_currentTrack.VibeShortName}"
            : "";
        NowPlayingSubtitle.Text = _currentTrack.Artist + suffix;
        MiniCoverMonogram.Text = _currentTrack.Monogram;

        var accentColor = ParseColor(_currentTrack.VibeColorHex, Color.FromRgb(99, 102, 241));
        MiniCoverBorder.Background = CreateFrozenBrush(Color.FromArgb(48, accentColor.R, accentColor.G, accentColor.B));
    }

    private void ShowTrackInInspector(TrackItem? track)
    {
        _inspectedTrack = track;

        if (track == null)
        {
            InspectorTitle.Text = "Трек не выбран";
            InspectorArtist.Text = "Выберите трек для просмотра AI-анализа";
            InspectorMetaLine.Text = string.Empty;
            ArtworkMonogram.Text = "♪";
            ArtworkImage.Visibility = Visibility.Collapsed;
            InspectorVibeCard.Visibility = Visibility.Collapsed;
            InspectorVibesSection.Visibility = Visibility.Collapsed;
            InspectorGenresSection.Visibility = Visibility.Collapsed;
            InspectorErrorBox.Visibility = Visibility.Collapsed;
            return;
        }

        InspectorTitle.Text = track.Title;
        InspectorArtist.Text = track.Artist;
        string bpmPart = !string.IsNullOrEmpty(track.BpmEnergyText) ? $"  ·  {track.BpmEnergyText}" : "";
        InspectorMetaLine.Text = $"{track.Format}  ·  {track.DurationText}" +
            (track.Status == TrackStatus.Ready ? bpmPart : $"  ·  {track.StatusLabel}");
        ArtworkMonogram.Text = track.Monogram;

        // Tint artwork backdrop with the track's vibe color
        var tint = ParseColor(track.VibeColorHex, Color.FromRgb(99, 102, 241));
        var grad = new LinearGradientBrush(
            Color.FromArgb(115, tint.R, tint.G, tint.B),
            Color.FromRgb(18, 22, 34),
            new Point(0, 0),
            new Point(1, 1));
        grad.Freeze();
        ArtworkGradientBg.Background = grad;

        LoadCoverArtAsync(track);

        if (track.Status == TrackStatus.Ready)
        {
            InspectorErrorBox.Visibility = Visibility.Collapsed;
            InspectorVibeCard.Visibility = Visibility.Visible;
            InspectorVibesSection.Visibility = Visibility.Visible;
            InspectorGenresSection.Visibility = Visibility.Visible;

            var vibeBrush = CreateFrozenBrush(track.VibeColorHex);
            InspectorVibeDot.Fill = vibeBrush;
            InspectorVibeLabel.Text = track.Vibe;
            InspectorVibeLabel.Foreground = vibeBrush;
            InspectorVibePct.Text = track.VibeConfidenceText;
            InspectorVibePct.Foreground = vibeBrush;
            InspectorVibeCard.BorderBrush = CreateFrozenBrush(Color.FromArgb(90, tint.R, tint.G, tint.B));

            InspectorVibesList.ItemsSource = track.VibeScores;
            InspectorGenresList.ItemsSource = track.TopGenres;
        }
        else if (track.Status == TrackStatus.Error)
        {
            InspectorVibeCard.Visibility = Visibility.Collapsed;
            InspectorVibesSection.Visibility = Visibility.Collapsed;
            InspectorGenresSection.Visibility = Visibility.Collapsed;
            InspectorErrorBox.Visibility = Visibility.Visible;
            InspectorErrorText.Text = $"Ошибка анализа: {track.ErrorMessage}";
        }
        else
        {
            InspectorVibeCard.Visibility = Visibility.Collapsed;
            InspectorVibesSection.Visibility = Visibility.Collapsed;
            InspectorGenresSection.Visibility = Visibility.Collapsed;
            InspectorErrorBox.Visibility = Visibility.Collapsed;
        }
    }

    private async void LoadCoverArtAsync(TrackItem track)
    {
        _coverCts?.Cancel();
        var cts = new CancellationTokenSource();
        _coverCts = cts;

        ArtworkImage.Visibility = Visibility.Collapsed;
        if (_currentTrack == track)
            MiniCoverImage.Visibility = Visibility.Collapsed;

        BitmapImage? bmp = await Task.Run(() =>
        {
            if (cts.Token.IsCancellationRequested) return null;
            byte[]? bytes = AudioDecoder.ReadCoverArtBytes(track.FilePath);
            if (bytes == null || bytes.Length == 0) return null;

            try
            {
                using var ms = new MemoryStream(bytes);
                var image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.DecodePixelWidth = 320;
                image.StreamSource = ms;
                image.EndInit();
                image.Freeze(); // Safe for cross-thread UI binding
                return image;
            }
            catch
            {
                return null;
            }
        });

        if (cts.Token.IsCancellationRequested || _isClosing) return;

        if (bmp != null)
        {
            if (_inspectedTrack == track)
            {
                ArtworkImage.Source = bmp;
                ArtworkImage.Visibility = Visibility.Visible;
            }
            if (_currentTrack == track)
            {
                MiniCoverImage.Source = bmp;
                MiniCoverImage.Visibility = Visibility.Visible;
            }
        }
    }

    // ==================== UI EVENT HANDLERS ====================

    private async void AddFiles_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Выберите аудиофайлы",
            Multiselect = true,
            Filter = "Аудиофайлы|*.mp3;*.wav;*.flac;*.ogg;*.m4a;*.aac;*.wma;*.aiff;*.aif|Все файлы|*.*"
        };

        if (dlg.ShowDialog(this) == true && dlg.FileNames.Length > 0)
        {
            await AddPathsAsync(dlg.FileNames);
        }
    }

    private async void AddFolder_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Выберите папку с музыкой",
            Multiselect = false
        };

        if (dlg.ShowDialog(this) == true && !string.IsNullOrWhiteSpace(dlg.FolderName))
        {
            await AddPathsAsync([dlg.FolderName]);
        }
    }

    private void FilterItem_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: CategoryFilterItem clicked }) return;

        ShowLibraryView();

        foreach (var f in _libraryFilters) f.IsSelected = false;
        foreach (var f in _vibeFilters) f.IsSelected = false;
        foreach (var f in _genreFilters) f.IsSelected = false;

        clicked.IsSelected = true;
        _activeFilter = clicked;
        RefreshCountsAndFilter();
    }

    private void SearchBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        SearchPlaceholder.Visibility = string.IsNullOrEmpty(SearchBox.Text)
            ? Visibility.Visible
            : Visibility.Collapsed;
        RefreshCountsAndFilter();
    }

    private void PlayFiltered_Click(object sender, RoutedEventArgs e)
    {
        if (_visibleTracks.Count > 0)
            PlayTrack(_visibleTracks[0]);
    }

    private void TrackListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (TrackListBox.SelectedItem is TrackItem selected)
            ShowTrackInInspector(selected);
    }

    private void TrackListBox_MouseDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (TrackListBox.SelectedItem is TrackItem selected)
            PlayTrack(selected);
    }

    private void PlayPause_Click(object sender, RoutedEventArgs e)
    {
        if (!_player.HasTrack)
        {
            if (TrackListBox.SelectedItem is TrackItem selected)
                PlayTrack(selected);
            else if (_visibleTracks.Count > 0)
                PlayTrack(_visibleTracks[0]);
            return;
        }

        _player.TogglePlayPause();
    }

    private void Next_Click(object sender, RoutedEventArgs e) => PlayNextTrack();
    private void Prev_Click(object sender, RoutedEventArgs e) => PlayPreviousTrack();

    private void Shuffle_Click(object sender, RoutedEventArgs e)
    {
        _isShuffle = !_isShuffle;
        ShuffleBtn.Foreground = _isShuffle
            ? CreateFrozenBrush("#818CF8")
            : CreateFrozenBrush("#94A3B8");
    }

    private void Repeat_Click(object sender, RoutedEventArgs e)
    {
        _isRepeat = !_isRepeat;
        RepeatBtn.Foreground = _isRepeat
            ? CreateFrozenBrush("#818CF8")
            : CreateFrozenBrush("#94A3B8");
    }

    private void Mute_Click(object sender, RoutedEventArgs e)
    {
        _player.IsMuted = !_player.IsMuted;
        UpdateVolumeIcon();
    }

    private void VolumeSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!IsLoaded) return;
        float vol = (float)(e.NewValue / 100.0);
        _player.Volume = vol;
        if (_player.IsMuted && vol > 0)
            _player.IsMuted = false;

        VolumePctText.Text = $"{(int)Math.Round(e.NewValue)}%";
        UpdateVolumeIcon();
    }

    private void UpdateVolumeIcon()
    {
        bool muted = _player.IsMuted || _player.Volume <= 0.001f;
        VolOnPath.Visibility = muted ? Visibility.Collapsed : Visibility.Visible;
        VolMutePath.Visibility = muted ? Visibility.Visible : Visibility.Collapsed;
    }

    private void SeekSlider_PreviewMouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!_player.HasTrack || _player.Duration.TotalSeconds <= 0)
            return;

        e.Handled = true;
        _isUserSeeking = true;
        SeekSlider.CaptureMouse();
        UpdateSeekSliderFromPoint(e.GetPosition(SeekSlider));
    }

    private void SeekSlider_PreviewMouseMove(object sender, MouseEventArgs e)
    {
        if (!_isUserSeeking) return;

        if (e.LeftButton == MouseButtonState.Pressed)
        {
            UpdateSeekSliderFromPoint(e.GetPosition(SeekSlider));
        }
        else
        {
            FinishUserSeek();
        }
    }

    private void SeekSlider_PreviewMouseUp(object sender, MouseButtonEventArgs e)
    {
        if (!_isUserSeeking) return;

        e.Handled = true;
        UpdateSeekSliderFromPoint(e.GetPosition(SeekSlider));
        FinishUserSeek();
    }

    private void SeekSlider_LostMouseCapture(object sender, MouseEventArgs e)
    {
        if (_isUserSeeking)
        {
            FinishUserSeek();
        }
    }

    private void UpdateSeekSliderFromPoint(Point pt)
    {
        double width = Math.Max(1.0, SeekSlider.ActualWidth);
        double ratio = Math.Clamp(pt.X / width, 0.0, 1.0);
        double targetSec = ratio * SeekSlider.Maximum;
        SeekSlider.Value = targetSec;
        TimeCurrentText.Text = FormatTime(TimeSpan.FromSeconds(targetSec));
    }

    private void FinishUserSeek()
    {
        if (!_isUserSeeking) return;
        _isUserSeeking = false;

        if (SeekSlider.IsMouseCaptured)
            SeekSlider.ReleaseMouseCapture();

        if (_player.HasTrack && _player.Duration.TotalSeconds > 0)
        {
            double targetSec = SeekSlider.Value;
            double totalSec = _player.Duration.TotalSeconds;

            // If dragged to the end of the track, immediately advance to next track (or repeat)
            if (targetSec >= totalSec - 0.4)
            {
                OnTrackEnded();
                return;
            }

            _player.Seek(TimeSpan.FromSeconds(targetSec));
        }

        UpdateSeekUi();
    }

    private void ContextPlay_Click(object sender, RoutedEventArgs e)
    {
        if (TrackListBox.SelectedItem is TrackItem t) PlayTrack(t);
    }

    private void ContextReanalyze_Click(object sender, RoutedEventArgs e)
    {
        if (TrackListBox.SelectedItem is TrackItem t)
            EnqueueForAnalysis([t]);
    }

    private void ContextShowInExplorer_Click(object sender, RoutedEventArgs e)
    {
        if (TrackListBox.SelectedItem is TrackItem t && File.Exists(t.FilePath))
        {
            Process.Start("explorer.exe", $"/select,\"{t.FilePath}\"");
        }
    }

    private void ContextRemove_Click(object sender, RoutedEventArgs e)
    {
        if (TrackListBox.SelectedItem is not TrackItem t) return;

        if (_currentTrack == t)
        {
            _player.Stop();
            _currentTrack = null;
            UpdateNowPlayingBar();
        }
        if (_inspectedTrack == t)
        {
            ShowTrackInInspector(null);
        }

        _allTracks.Remove(t);
        _trackPathIndex.Remove(t.FilePath);
        RefreshCountsAndFilter();

        var snapshot = _allTracks.ToList();
        _ = Task.Run(() => LibraryCache.Save(snapshot));
    }

    private async void ExportFolders_Click(object sender, RoutedEventArgs e)
    {
        var readyTracks = _allTracks.Where(t => t.Status == TrackStatus.Ready && File.Exists(t.FilePath)).ToList();
        if (readyTracks.Count == 0)
        {
            MessageBox.Show("Нет проанализированных треков для сортировки по папкам.", "SoundPulse AI",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var dlg = new OpenFolderDialog
        {
            Title = "Выберите папку, куда скопировать треки по категориям (Вайб / Жанр)"
        };

        if (dlg.ShowDialog(this) != true || string.IsNullOrWhiteSpace(dlg.FolderName))
            return;

        string targetRoot = dlg.FolderName;
        int copied = await Task.Run(() =>
        {
            int count = 0;
            foreach (var t in readyTracks)
            {
                string safeVibe = SanitizeFolderName(t.VibeShortName);
                string safeGenre = SanitizeFolderName(t.GenreDisplay);
                string destDir = Path.Combine(targetRoot, safeVibe, safeGenre);
                Directory.CreateDirectory(destDir);

                string destFile = Path.Combine(destDir, Path.GetFileName(t.FilePath));
                try
                {
                    File.Copy(t.FilePath, destFile, overwrite: true);
                    count++;
                }
                catch
                {
                    // Skip locked files
                }
            }
            return count;
        });

        MessageBox.Show(
            $"Готово! Скопировано треков: {copied}\nСтруктура: <Папка>\\<Вайб>\\<Жанр>\\<Файл>",
            "SoundPulse AI",
            MessageBoxButton.OK,
            MessageBoxImage.Information);
    }

    private void ClearLibrary_Click(object sender, RoutedEventArgs e)
    {
        if (_allTracks.Count == 0) return;

        var res = MessageBox.Show(
            "Очистить список треков в плеере? (Файлы на диске не удаляются)",
            "SoundPulse AI",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);

        if (res != MessageBoxResult.Yes) return;

        _player.Stop();
        _currentTrack = null;
        _allTracks.Clear();
        _trackPathIndex.Clear();
        ShowTrackInInspector(null);
        UpdateNowPlayingBar();
        RefreshCountsAndFilter();
        _ = Task.Run(() => LibraryCache.Save([]));
    }

    private void Window_DragEnter(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            e.Effects = DragDropEffects.Copy;
            DragOverlay.Visibility = Visibility.Visible;
        }
        else
        {
            e.Effects = DragDropEffects.None;
        }
    }

    private void Window_DragLeave(object sender, DragEventArgs e)
    {
        DragOverlay.Visibility = Visibility.Collapsed;
    }

    private async void Window_Drop(object sender, DragEventArgs e)
    {
        DragOverlay.Visibility = Visibility.Collapsed;
        if (e.Data.GetData(DataFormats.FileDrop) is string[] paths && paths.Length > 0)
        {
            await AddPathsAsync(paths);
        }
    }

    private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (SearchBox.IsKeyboardFocused || DownloadQueryBox.IsKeyboardFocused) return;

        if (e.Key == Key.Space)
        {
            PlayPause_Click(sender, e);
            e.Handled = true;
        }
        else if (e.Key == Key.Right && _player.HasTrack)
        {
            _player.Seek(_player.Position + TimeSpan.FromSeconds(5));
            UpdateSeekUi();
            e.Handled = true;
        }
        else if (e.Key == Key.Left && _player.HasTrack)
        {
            _player.Seek(_player.Position - TimeSpan.FromSeconds(5));
            UpdateSeekUi();
            e.Handled = true;
        }
        else if (e.Key == Key.O && Keyboard.Modifiers == ModifierKeys.Control)
        {
            AddFiles_Click(sender, e);
            e.Handled = true;
        }
    }

    private void Window_StateChanged(object? sender, EventArgs e)
    {
        if (WindowState == WindowState.Maximized)
        {
            RootGrid.Margin = new Thickness(7);
            MaximizeIconPath.Data = Geometry.Parse("M 2.5,0.5 H 9.5 V 7.5 M 0.5,2.5 H 7.5 V 9.5 H 0.5 Z");
        }
        else
        {
            RootGrid.Margin = new Thickness(0);
            MaximizeIconPath.Data = Geometry.Parse("M 1,1 H 9 V 9 H 1 Z");
        }
    }

    private void Minimize_Click(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

    private void Maximize_Click(object sender, RoutedEventArgs e)
        => WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;

    private void Close_Click(object sender, RoutedEventArgs e) => Close();

    private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        _isClosing = true;
        _playbackTimer.Stop();
        _coverCts?.Cancel();
        _coverCts?.Dispose();
        _searchCts?.Cancel();
        _searchCts?.Dispose();
        _settings.Save();
        LibraryCache.Save(_allTracks, CurrentCacheVersion);
        _watchFolder.Dispose();
        _downloader.Dispose();
        _player.Dispose();
        _analysisWorker?.Dispose();
        _classifier?.Dispose();
    }

    // ==================== DOWNLOAD HUB & WATCH FOLDER ====================

    private void ApplySettingsToUi()
    {
        _isSyncingSettings = true;
        try
        {
            DownloadFolderPathText.Text = _settings.DownloadFolder;
            AutoSortCheckBox.IsChecked = _settings.AutoSortByCategory;
            WatchFolderCheckBox.IsChecked = _settings.WatchFolderEnabled;
            WatchFolderPathText.Text = string.IsNullOrWhiteSpace(_settings.WatchFolderPath)
                ? "Не выбрана (например, Загрузки или папка Telegram Desktop)"
                : _settings.WatchFolderPath;

            bool active = _watchFolder.Configure(_settings.WatchFolderEnabled, _settings.WatchFolderPath);
            WatchStatusBadge.Text = active ? "● Активен" : (_settings.WatchFolderEnabled ? "Папка не найдена" : "");
            WatchStatusBadge.Foreground = active ? CreateFrozenBrush("#10B981") : CreateFrozenBrush("#FBBF24");
            SidebarWatchDot.Visibility = active ? Visibility.Visible : Visibility.Collapsed;
        }
        finally
        {
            _isSyncingSettings = false;
        }
    }

    private void OnWatchFolderAudioArrived(string filePath)
    {
        Dispatcher.BeginInvoke(async () =>
        {
            if (_isClosing) return;
            await AddPathsAsync([filePath], markForAutoSort: false);
        });
    }

    private void TryAutoSortTrackFile(TrackItem track)
    {
        try
        {
            if (!File.Exists(track.FilePath)) return;
            string destDir = Path.Combine(
                _settings.DownloadFolder,
                SanitizeFolderName(track.VibeShortName),
                SanitizeFolderName(track.GenreDisplay));
            Directory.CreateDirectory(destDir);

            string destPath = Path.Combine(destDir, Path.GetFileName(track.FilePath));
            if (string.Equals(destPath, track.FilePath, StringComparison.OrdinalIgnoreCase))
                return;

            if (File.Exists(destPath))
            {
                string name = Path.GetFileNameWithoutExtension(track.FilePath);
                string ext = Path.GetExtension(track.FilePath);
                for (int i = 2; i < 1000; i++)
                {
                    string candidate = Path.Combine(destDir, $"{name} ({i}){ext}");
                    if (!File.Exists(candidate))
                    {
                        destPath = candidate;
                        break;
                    }
                }
            }

            string oldPath = track.FilePath;
            File.Move(oldPath, destPath);
            _trackPathIndex.Remove(oldPath);
            _trackPathIndex.Add(destPath);
            track.FilePath = destPath;
        }
        catch
        {
            // Non-fatal: keep file in place if locked
        }
    }

    private void OpenDownloadView_Click(object sender, RoutedEventArgs e)
    {
        LibraryViewGrid.Visibility = Visibility.Collapsed;
        DownloadViewGrid.Visibility = Visibility.Visible;
        DownloadTabBtn.BorderBrush = CreateFrozenBrush("#6366F1");
        DownloadQueryBox.Focus();
    }

    private void BackToLibrary_Click(object sender, RoutedEventArgs e) => ShowLibraryView();

    private void ShowLibraryView()
    {
        DownloadViewGrid.Visibility = Visibility.Collapsed;
        LibraryViewGrid.Visibility = Visibility.Visible;
        DownloadTabBtn.ClearValue(BorderBrushProperty);
    }

    private void DownloadQueryBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        DownloadQueryPlaceholder.Visibility = string.IsNullOrEmpty(DownloadQueryBox.Text)
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    private void DownloadQueryBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            e.Handled = true;
            DownloadSearchOrUrl_Click(sender, e);
        }
    }

    private async void DownloadSearchOrUrl_Click(object sender, RoutedEventArgs e)
    {
        string input = DownloadQueryBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(input)) return;

        _searchCts?.Cancel();
        _searchCts = new CancellationTokenSource();
        var ct = _searchCts.Token;

        DownloadActionBtn.IsEnabled = false;
        DownloadAllPlaylistBtn.Visibility = Visibility.Collapsed;

        try
        {
            if (MusicDownloader.IsHttpUrl(input))
            {
                // Case 1: Direct audio URL (.mp3, .m4a, .flac, .ogg, .wav) -> download via HttpClient
                if (MusicDownloader.LooksLikeDirectAudioUrl(input))
                {
                    ShowDownloadStatus("Скачивание аудиофайла по прямой ссылке…", 0, indeterminate: false);
                    var progress = new Progress<double>(p =>
                    {
                        DownloadStatusBar.IsIndeterminate = false;
                        DownloadStatusBar.Value = p;
                        DownloadStatusText.Text = $"Скачивание по ссылке… {(int)Math.Round(p)}%";
                    });

                    string savedPath = await _downloader.DownloadDirectAudioUrlAsync(input, _settings.DownloadFolder, progress, ct);
                    DownloadQueryBox.Text = string.Empty;
                    ShowDownloadStatus($"Скачано: {Path.GetFileName(savedPath)} · Запущен AI-анализ", 100, indeterminate: false);
                    var addedDirect = await AddPathsAsync([savedPath], markForAutoSort: _settings.AutoSortByCategory);
                    FocusTracksInLibrary(addedDirect);
                    return;
                }

                // Case 2: Web page URL (YouTube / SoundCloud / Bandcamp / etc.) -> auto-ensure yt-dlp.exe & download!
                string? ytDlp = MusicDownloader.FindYtDlpExecutable();
                if (ytDlp == null)
                {
                    ShowDownloadStatus("Первая загрузка с сайта: скачиваю движок yt-dlp (один раз)…", 0, indeterminate: false);
                    var dlpProgress = new Progress<double>(p =>
                    {
                        DownloadStatusBar.IsIndeterminate = false;
                        DownloadStatusBar.Value = p;
                        DownloadStatusText.Text = $"Установка yt-dlp (1 раз)… {(int)Math.Round(p)}%";
                    });
                    ytDlp = await _downloader.EnsureYtDlpAsync(dlpProgress, ct);
                }

                ShowDownloadStatus("Получение аудиопотока…", 0, indeterminate: true);
                var ytProgress = new Progress<double>(p =>
                {
                    DownloadStatusBar.IsIndeterminate = false;
                    DownloadStatusBar.Value = p;
                    DownloadStatusText.Text = $"Скачивание аудио… {(int)Math.Round(p)}%";
                });

                var downloadedFiles = await _downloader.DownloadViaYtDlpAsync(ytDlp, input, _settings.DownloadFolder, ytProgress, ct);
                if (downloadedFiles.Count > 0)
                {
                    DownloadQueryBox.Text = string.Empty;
                    ShowDownloadStatus($"Скачано треков: {downloadedFiles.Count} · Запущен AI-анализ!", 100, indeterminate: false);
                    var addedYt = await AddPathsAsync(downloadedFiles, markForAutoSort: _settings.AutoSortByCategory);
                    if (addedYt.Count > 0)
                    {
                        FocusTracksInLibrary(addedYt);
                    }
                    else
                    {
                        ShowDownloadStatus(
                            $"Файл скачан ({Path.GetFileName(downloadedFiles[0])}), но его формат не поддерживается плеером.",
                            0,
                            indeterminate: false);
                    }
                }
                else
                {
                    ShowDownloadStatus("Загрузка завершилась, но аудиофайл не найден.", 0, indeterminate: false);
                }
                return;
            }

            // Case 3: Text search query -> search open Audius catalog (no API key, no region locks)
            ShowDownloadStatus($"Поиск «{input}» в каталоге Audius…", 25, indeterminate: true);
            var results = await _downloader.SearchAudiusAsync(input, 15, ct);
            _searchResults.Clear();
            foreach (var r in results) _searchResults.Add(r);

            DownloadEmptyHint.Visibility = _searchResults.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
            ShowDownloadStatus(
                results.Count > 0
                    ? $"Найдено: {results.Count}. Нажмите «Скачать» на нужном треке."
                    : "Ничего не найдено. Попробуйте другой запрос.",
                100,
                indeterminate: false);
        }
        catch (OperationCanceledException)
        {
            // Superseded by new search
        }
        catch (Exception ex)
        {
            ShowDownloadStatus($"Ошибка: {ex.Message}", 0, indeterminate: false);
        }
        finally
        {
            DownloadActionBtn.IsEnabled = true;
        }
    }

    private async void DownloadResultItem_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: SearchResultItem item }) return;
        await DownloadSingleResultAsync(item, CancellationToken.None);
    }

    private async void DownloadAllResults_Click(object sender, RoutedEventArgs e)
    {
        var pending = _searchResults.Where(r => r.CanDownload).ToList();
        if (pending.Count == 0) return;

        DownloadAllPlaylistBtn.IsEnabled = false;
        try
        {
            int done = 0;
            foreach (var item in pending)
            {
                if (_isClosing) break;
                ShowDownloadStatus($"Плейлист ({done + 1}/{pending.Count}): {item.Artist} — {item.Title}", 100.0 * done / pending.Count, false);
                await DownloadSingleResultAsync(item, CancellationToken.None);
                done++;
            }
            ShowDownloadStatus($"Загрузка плейлиста завершена ({done} треков)!", 100, false);
        }
        finally
        {
            DownloadAllPlaylistBtn.IsEnabled = true;
        }
    }

    private async Task DownloadSingleResultAsync(SearchResultItem item, CancellationToken ct)
    {
        if (!item.CanDownload) return;

        item.IsDownloading = true;
        item.DownloadProgress = 0;
        ShowDownloadStatus($"Скачивание: {item.Artist} — {item.Title}…", 0, indeterminate: false);

        try
        {
            var progress = new Progress<double>(p =>
            {
                item.DownloadProgress = p;
                DownloadStatusBar.IsIndeterminate = false;
                DownloadStatusBar.Value = p;
                DownloadStatusText.Text = $"Скачивание: {item.Title} ({(int)Math.Round(p)}%)";
            });

            string savedPath = await _downloader.DownloadSearchResultAsync(item, _settings.DownloadFolder, progress, ct);
            item.IsDownloaded = true;
            ShowDownloadStatus($"Добавлено в библиотеку: {item.Artist} — {item.Title} · AI-анализ…", 100, indeterminate: false);

            var added = await AddPathsAsync([savedPath], markForAutoSort: _settings.AutoSortByCategory);
            if (added.Count > 0)
                ShowTrackInInspector(added[^1]);
        }
        catch (Exception ex)
        {
            ShowDownloadStatus($"Не удалось скачать «{item.Title}»: {ex.Message}", 0, indeterminate: false);
        }
        finally
        {
            item.IsDownloading = false;
        }
    }

    private void ShowDownloadStatus(string text, double percent, bool indeterminate)
    {
        DownloadStatusBanner.Visibility = Visibility.Visible;
        DownloadStatusText.Text = text;
        DownloadStatusBar.IsIndeterminate = indeterminate;
        if (!indeterminate)
            DownloadStatusBar.Value = Math.Clamp(percent, 0, 100);
    }

    private void ChangeDownloadFolder_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Выберите папку для сохранения скачанных треков",
            InitialDirectory = Directory.Exists(_settings.DownloadFolder) ? _settings.DownloadFolder : string.Empty
        };

        if (dlg.ShowDialog(this) == true && !string.IsNullOrWhiteSpace(dlg.FolderName))
        {
            _settings.DownloadFolder = dlg.FolderName;
            _settings.Save();
            ApplySettingsToUi();
        }
    }

    private void OpenDownloadFolder_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            Directory.CreateDirectory(_settings.DownloadFolder);
            Process.Start("explorer.exe", $"\"{_settings.DownloadFolder}\"");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Не удалось открыть папку:\n{ex.Message}", "SoundPulse AI", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void AutoSortCheckBox_Changed(object sender, RoutedEventArgs e)
    {
        if (_isSyncingSettings) return;
        _settings.AutoSortByCategory = AutoSortCheckBox.IsChecked == true;
        _settings.Save();
    }

    private void WatchFolderCheckBox_Changed(object sender, RoutedEventArgs e)
    {
        if (_isSyncingSettings) return;
        bool wantEnable = WatchFolderCheckBox.IsChecked == true;

        if (wantEnable && (string.IsNullOrWhiteSpace(_settings.WatchFolderPath) || !Directory.Exists(_settings.WatchFolderPath)))
        {
            ChooseWatchFolder_Click(sender, e);
            return;
        }

        _settings.WatchFolderEnabled = wantEnable;
        _settings.Save();
        ApplySettingsToUi();
    }

    private void ChooseWatchFolder_Click(object sender, RoutedEventArgs e)
    {
        string downloads = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        var dlg = new OpenFolderDialog
        {
            Title = "Выберите папку автоимпорта (например, Загрузки или Telegram Desktop)",
            InitialDirectory = Directory.Exists(_settings.WatchFolderPath)
                ? _settings.WatchFolderPath
                : (Directory.Exists(downloads) ? downloads : string.Empty)
        };

        if (dlg.ShowDialog(this) == true && !string.IsNullOrWhiteSpace(dlg.FolderName))
        {
            _settings.WatchFolderPath = dlg.FolderName;
            _settings.WatchFolderEnabled = true;
            _settings.Save();
        }
        else if (string.IsNullOrWhiteSpace(_settings.WatchFolderPath))
        {
            _settings.WatchFolderEnabled = false;
        }

        ApplySettingsToUi();
    }

    // ==================== HELPERS ====================

    private static string FormatTime(TimeSpan ts)
    {
        if (ts < TimeSpan.Zero) ts = TimeSpan.Zero;
        return ts.TotalHours >= 1
            ? $"{(int)ts.TotalHours}:{ts.Minutes:D2}:{ts.Seconds:D2}"
            : $"{ts.Minutes:D2}:{ts.Seconds:D2}";
    }

    private static string PluralizeTracks(int n)
    {
        int mod100 = n % 100;
        if (mod100 is >= 11 and <= 14) return "треков";
        return (n % 10) switch
        {
            1 => "трек",
            2 or 3 or 4 => "трека",
            _ => "треков"
        };
    }

    private static string SanitizeFolderName(string name)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return string.IsNullOrWhiteSpace(name) ? "Other" : name.Trim();
    }

    private static Color ParseColor(string hex, Color fallback)
    {
        try { return (Color)ColorConverter.ConvertFromString(hex); }
        catch { return fallback; }
    }

    private static SolidColorBrush CreateFrozenBrush(string hex)
        => CreateFrozenBrush(ParseColor(hex, Color.FromRgb(129, 140, 248)));

    private static SolidColorBrush CreateFrozenBrush(Color c)
    {
        var b = new SolidColorBrush(c);
        b.Freeze();
        return b;
    }
}
