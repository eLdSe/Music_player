using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;

namespace SoundPulse.Domain;

public sealed class ModelLabels
{
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = "MusicAudioNet";

    [JsonPropertyName("version")]
    public int Version { get; set; } = 1;

    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; set; } = 16000;

    [JsonPropertyName("chunk_duration_sec")]
    public double ChunkDurationSec { get; set; } = 10.0;

    [JsonPropertyName("chunk_samples")]
    public int ChunkSamples { get; set; } = 160000;

    [JsonPropertyName("n_mels")]
    public int NMels { get; set; } = 128;

    [JsonPropertyName("n_fft")]
    public int NFft { get; set; } = 1024;

    [JsonPropertyName("hop_length")]
    public int HopLength { get; set; } = 512;

    [JsonPropertyName("genres")]
    public List<string> Genres { get; set; } = [];

    [JsonPropertyName("vibes")]
    public List<string> Vibes { get; set; } = [];
}

public sealed record ScoreEntry(
    string Key,
    string DisplayName,
    float Probability,
    string AccentHex)
{
    public double PercentValue => Math.Round(Probability * 100.0, 1);
    public string PercentText => $"{Probability * 100f:0.0}%";
}

public sealed class AnalysisResult
{
    public required string Genre { get; init; }
    public required float GenreConfidence { get; init; }
    public required string Vibe { get; init; }
    public required int VibeIndex { get; init; }
    public required float VibeConfidence { get; init; }
    public required List<ScoreEntry> TopGenres { get; init; }
    public required List<ScoreEntry> VibeScores { get; init; }
    public required double DurationSec { get; init; }
    public required double InferenceMs { get; init; }
    public int Bpm { get; init; }
    public float Energy { get; init; }
}

public enum TrackStatus
{
    Pending,
    Analyzing,
    Ready,
    Error
}

public static class CategoryPalette
{
    public static string GetVibeColor(int vibeIndex, string? vibeName = null)
    {
        if (!string.IsNullOrWhiteSpace(vibeName))
        {
            string lower = vibeName.ToLowerInvariant();
            if (lower.Contains("драйв") || lower.Contains("энерг")) return "#F43F5E"; // Crimson Coral
            if (lower.Contains("вечерин") || lower.Contains("танц")) return "#F59E0B"; // Amber Gold
            if (lower.Contains("чилл") || lower.Contains("отдых")) return "#10B981"; // Emerald Mint
            if (lower.Contains("лирик") || lower.Contains("груст")) return "#A855F7"; // Violet Dusk
            if (lower.Contains("фокус") || lower.Contains("концентр")) return "#38BDF8"; // Cyan Sky
        }

        return vibeIndex switch
        {
            0 => "#F43F5E",
            1 => "#10B981",
            2 => "#38BDF8",
            3 => "#A855F7",
            4 => "#F59E0B",
            _ => "#6366F1"
        };
    }

    public static string GetVibeShortName(int vibeIndex, string fallback)
    {
        if (!string.IsNullOrWhiteSpace(fallback))
        {
            int slash = fallback.IndexOf('/');
            if (slash > 0) return fallback[..slash].Trim();
            return fallback.Trim();
        }

        return vibeIndex switch
        {
            0 => "Драйв",
            1 => "Чилл",
            2 => "Фокус",
            _ => "—"
        };
    }

    public static string GetVibeIcon(int vibeIndex) => "●";

    public static string FormatGenreName(string genre) => genre.ToLowerInvariant() switch
    {
        "blues" => "Blues",
        "classical" => "Classical",
        "country" => "Country",
        "disco" => "Disco",
        "electronic" => "Electronic",
        "folk" => "Folk",
        "hiphop" => "Hip-Hop",
        "jazz" => "Jazz",
        "lofi_ambient" => "Lo-Fi",
        "metal" => "Metal",
        "pop" => "Pop",
        "reggae" => "Reggae",
        "rock" => "Rock",
        _ => string.IsNullOrWhiteSpace(genre)
            ? "—"
            : char.ToUpperInvariant(genre[0]) + genre[1..]
    };

    public static string GetGenreColor(string genre) => genre.ToLowerInvariant() switch
    {
        "rock" => "#FB7185",
        "metal" => "#F43F5E",
        "hiphop" => "#F97316",
        "disco" => "#E879F9",
        "electronic" => "#E879F9",
        "pop" => "#34D399",
        "reggae" => "#2DD4BF",
        "blues" => "#60A5FA",
        "country" => "#FBBF24",
        "folk" => "#FACC15",
        "classical" => "#38BDF8",
        "jazz" => "#A78BFA",
        "lofi_ambient" => "#22D3EE",
        _ => "#818CF8"
    };
}

public sealed class TrackItem : INotifyPropertyChanged
{
    private string _title = string.Empty;
    private string _artist = "Неизвестный исполнитель";
    private string _album = string.Empty;
    private double _durationSec;
    private string _genre = string.Empty;
    private float _genreConfidence;
    private string _vibe = string.Empty;
    private int _vibeIndex = -1;
    private float _vibeConfidence;
    private TrackStatus _status = TrackStatus.Pending;
    private string _errorMessage = string.Empty;
    private bool _isPlaying;
    private double _inferenceMs;
    private int _bpm;
    private float _energy;
    private List<ScoreEntry> _topGenres = [];
    private List<ScoreEntry> _vibeScores = [];

    private string _filePath = string.Empty;
    public required string FilePath
    {
        get => _filePath;
        set
        {
            if (SetField(ref _filePath, value))
            {
                OnPropertyChanged(nameof(FileName));
                OnPropertyChanged(nameof(Format));
            }
        }
    }
    public string FileName => Path.GetFileName(_filePath);
    public string Format => Path.GetExtension(_filePath).TrimStart('.').ToUpperInvariant();
    public bool AutoSortPending { get; set; }
    public long FileSize { get; set; }
    public long LastWriteUtcTicks { get; set; }

    public string Title
    {
        get => _title;
        set { if (SetField(ref _title, value)) OnPropertyChanged(nameof(Monogram)); }
    }

    public string Artist
    {
        get => _artist;
        set => SetField(ref _artist, value);
    }

    public string Album
    {
        get => _album;
        set => SetField(ref _album, value);
    }

    public double DurationSec
    {
        get => _durationSec;
        set { if (SetField(ref _durationSec, value)) OnPropertyChanged(nameof(DurationText)); }
    }

    public string DurationText
    {
        get
        {
            if (_durationSec <= 0) return "--:--";
            var ts = TimeSpan.FromSeconds(_durationSec);
            return ts.TotalHours >= 1
                ? $"{(int)ts.TotalHours}:{ts.Minutes:D2}:{ts.Seconds:D2}"
                : $"{ts.Minutes:D2}:{ts.Seconds:D2}";
        }
    }

    public string Genre
    {
        get => _genre;
        set
        {
            if (SetField(ref _genre, value))
            {
                OnPropertyChanged(nameof(GenreDisplay));
                OnPropertyChanged(nameof(GenreColorHex));
                OnPropertyChanged(nameof(GenreBadgeText));
            }
        }
    }

    public float GenreConfidence
    {
        get => _genreConfidence;
        set
        {
            if (SetField(ref _genreConfidence, value))
            {
                OnPropertyChanged(nameof(GenreBadgeText));
                OnPropertyChanged(nameof(GenreConfidenceText));
            }
        }
    }

    public string Vibe
    {
        get => _vibe;
        set
        {
            if (SetField(ref _vibe, value))
            {
                OnPropertyChanged(nameof(VibeBadgeText));
                OnPropertyChanged(nameof(VibeShortName));
            }
        }
    }

    public int VibeIndex
    {
        get => _vibeIndex;
        set
        {
            if (SetField(ref _vibeIndex, value))
            {
                OnPropertyChanged(nameof(VibeColorHex));
                OnPropertyChanged(nameof(VibeIcon));
                OnPropertyChanged(nameof(VibeShortName));
                OnPropertyChanged(nameof(VibeBadgeText));
            }
        }
    }

    public float VibeConfidence
    {
        get => _vibeConfidence;
        set { if (SetField(ref _vibeConfidence, value)) OnPropertyChanged(nameof(VibeConfidenceText)); }
    }

    public TrackStatus Status
    {
        get => _status;
        set
        {
            if (SetField(ref _status, value))
            {
                OnPropertyChanged(nameof(IsAnalyzing));
                OnPropertyChanged(nameof(IsReady));
                OnPropertyChanged(nameof(HasError));
                OnPropertyChanged(nameof(StatusLabel));
                OnPropertyChanged(nameof(GenreBadgeText));
                OnPropertyChanged(nameof(VibeBadgeText));
            }
        }
    }

    public string ErrorMessage
    {
        get => _errorMessage;
        set => SetField(ref _errorMessage, value);
    }

    public bool IsPlaying
    {
        get => _isPlaying;
        set => SetField(ref _isPlaying, value);
    }

    public double InferenceMs
    {
        get => _inferenceMs;
        set { if (SetField(ref _inferenceMs, value)) OnPropertyChanged(nameof(InferenceText)); }
    }

    public int Bpm
    {
        get => _bpm;
        set { if (SetField(ref _bpm, value)) OnPropertyChanged(nameof(BpmEnergyText)); }
    }

    public float Energy
    {
        get => _energy;
        set { if (SetField(ref _energy, value)) OnPropertyChanged(nameof(BpmEnergyText)); }
    }

    public string BpmEnergyText => _bpm > 0 ? $"{_bpm} BPM · Энергия {(int)Math.Round(_energy * 100f)}%" : string.Empty;

    public List<ScoreEntry> TopGenres
    {
        get => _topGenres;
        set => SetField(ref _topGenres, value);
    }

    public List<ScoreEntry> VibeScores
    {
        get => _vibeScores;
        set => SetField(ref _vibeScores, value);
    }

    public bool IsAnalyzing => _status is TrackStatus.Pending or TrackStatus.Analyzing;
    public bool IsReady => _status == TrackStatus.Ready;
    public bool HasError => _status == TrackStatus.Error;

    public string StatusLabel => _status switch
    {
        TrackStatus.Pending => "В очереди…",
        TrackStatus.Analyzing => "AI-анализ…",
        TrackStatus.Error => "Ошибка",
        _ => string.Empty
    };

    public string GenreDisplay => string.IsNullOrEmpty(_genre) ? "—" : CategoryPalette.FormatGenreName(_genre);
    public string GenreColorHex => CategoryPalette.GetGenreColor(_genre);
    public string GenreConfidenceText => _genreConfidence > 0 ? $"{_genreConfidence * 100f:0}%" : "";
    public string GenreBadgeText => _status switch
    {
        TrackStatus.Pending => "В очереди",
        TrackStatus.Analyzing => "Анализ…",
        TrackStatus.Error => "Ошибка",
        _ => string.IsNullOrEmpty(_genre) ? "—" : $"{GenreDisplay}  {_genreConfidence * 100f:0}%"
    };

    public string VibeColorHex => CategoryPalette.GetVibeColor(_vibeIndex, _vibe);
    public string VibeIcon => CategoryPalette.GetVibeIcon(_vibeIndex);
    public string VibeShortName => CategoryPalette.GetVibeShortName(_vibeIndex, _vibe);
    public string VibeConfidenceText => _vibeConfidence > 0 ? $"{_vibeConfidence * 100f:0}%" : "";
    public string VibeBadgeText => _status == TrackStatus.Ready && _vibeIndex >= 0
        ? VibeShortName
        : "—";

    public string InferenceText => _inferenceMs > 0 ? $"{_inferenceMs:0.0} мс" : "кэш";

    public string Monogram
    {
        get
        {
            if (string.IsNullOrWhiteSpace(_title)) return "♪";
            foreach (var ch in _title)
            {
                if (char.IsLetterOrDigit(ch))
                    return char.ToUpperInvariant(ch).ToString();
            }
            return "♪";
        }
    }

    public void ApplyAnalysis(AnalysisResult result)
    {
        if (result.DurationSec > 0)
            DurationSec = result.DurationSec;
        Genre = result.Genre;
        GenreConfidence = result.GenreConfidence;
        Vibe = result.Vibe;
        VibeIndex = result.VibeIndex;
        VibeConfidence = result.VibeConfidence;
        TopGenres = result.TopGenres;
        VibeScores = result.VibeScores;
        InferenceMs = result.InferenceMs;
        Bpm = result.Bpm;
        Energy = result.Energy;
        ErrorMessage = string.Empty;
        Status = TrackStatus.Ready;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

public enum FilterKind
{
    All,
    Vibe,
    Genre
}

public sealed class CategoryFilterItem : INotifyPropertyChanged
{
    private int _count;
    private bool _isSelected;

    public required string Key { get; init; }
    public required FilterKind Kind { get; init; }
    public required string Label { get; init; }
    public required string Icon { get; init; }
    public required string AccentHex { get; init; }
    public int VibeIndex { get; init; } = -1;

    public int Count
    {
        get => _count;
        set => SetField(ref _count, value);
    }

    public bool IsSelected
    {
        get => _isSelected;
        set => SetField(ref _isSelected, value);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }
}
