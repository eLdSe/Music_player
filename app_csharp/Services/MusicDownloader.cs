using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using SoundPulse.Audio;

namespace SoundPulse.Services;

public sealed class SearchResultItem : INotifyPropertyChanged
{
    private bool _isDownloading;
    private bool _isDownloaded;
    private double _downloadProgress;

    public required string Id { get; init; }
    public required string Title { get; init; }
    public required string Artist { get; init; }
    public required string StreamUrl { get; init; }
    public required string ThumbnailUrl { get; init; }
    public required string SourceLabel { get; init; }
    public TimeSpan? Duration { get; init; }

    public string DurationText => Duration is { TotalSeconds: > 0 } d
        ? (d.TotalHours >= 1 ? $"{(int)d.TotalHours}:{d.Minutes:D2}:{d.Seconds:D2}" : $"{d.Minutes:D2}:{d.Seconds:D2}")
        : "--:--";

    public bool IsDownloading
    {
        get => _isDownloading;
        set
        {
            if (SetField(ref _isDownloading, value))
            {
                OnPropertyChanged(nameof(CanDownload));
                OnPropertyChanged(nameof(ButtonText));
            }
        }
    }

    public bool IsDownloaded
    {
        get => _isDownloaded;
        set
        {
            if (SetField(ref _isDownloaded, value))
            {
                OnPropertyChanged(nameof(CanDownload));
                OnPropertyChanged(nameof(ButtonText));
            }
        }
    }

    public double DownloadProgress
    {
        get => _downloadProgress;
        set
        {
            if (SetField(ref _downloadProgress, value))
                OnPropertyChanged(nameof(ButtonText));
        }
    }

    public bool CanDownload => !_isDownloading && !_isDownloaded;

    public string ButtonText => _isDownloaded
        ? "В библиотеке"
        : _isDownloading
            ? $"{(int)Math.Round(_downloadProgress)}%"
            : "Скачать";

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

/// <summary>
/// Pure C# (HttpClient + System.Text.Json) music search & downloader with zero region locks:
/// 1. Audius open public API search & full-length MP3 download (with cover art + ID3 tags).
/// 2. Direct HTTP/HTTPS audio URL download (.mp3, .m4a, .flac, .ogg, .wav, .aac).
/// 3. Optional external yt-dlp.exe integration (auto-detected next to SoundPulse.exe or in PATH)
///    for YouTube, SoundCloud, Bandcamp, etc.
/// </summary>
public sealed class MusicDownloader : IDisposable
{
    private const string AudiusApiBase = "https://api.audius.co/v1";
    private const string AudiusAppName = "SoundPulse";
    private const string YtDlpDownloadUrl = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe";

    private static readonly Regex YtDlpPercentRegex = new(
        @"\[download\]\s+(\d+(?:\.\d+)?)%",
        RegexOptions.Compiled);

    private static readonly Regex NoiseSuffixRegex = new(
        @"\s*[\(\[]\s*(official\s*(music\s*)?(video|audio|lyric\s*video|visualizer)|lyrics?|audio|hd|hq|4k|remastered(\s*\d{4})?|explicit|clean|music\s*video)\s*[\)\]]",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private readonly HttpClient _http;

    public MusicDownloader()
    {
        var handler = new HttpClientHandler
        {
            AllowAutoRedirect = true,
            AutomaticDecompression = System.Net.DecompressionMethods.All
        };
        _http = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("SoundPulse/1.0 (+https://localhost)");
    }

    public static bool IsHttpUrl(string input)
        => Uri.TryCreate(input.Trim(), UriKind.Absolute, out var uri)
           && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);

    public static bool LooksLikeDirectAudioUrl(string url)
    {
        if (!Uri.TryCreate(url.Trim(), UriKind.Absolute, out var uri)) return false;
        string ext = Path.GetExtension(uri.AbsolutePath);
        return !string.IsNullOrWhiteSpace(ext) && AudioDecoder.SupportedExtensions.Contains(ext);
    }

    public static string ManagedYtDlpPath
    {
        get
        {
            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SoundPulse");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "yt-dlp.exe");
        }
    }

    public static string? FindYtDlpExecutable()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "yt-dlp.exe"),
            ManagedYtDlpPath
        ];

        foreach (string path in candidates)
        {
            if (File.Exists(path) && new FileInfo(path).Length > 100_000)
                return path;
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (!string.IsNullOrWhiteSpace(pathEnv))
        {
            foreach (string dir in pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            {
                try
                {
                    string p = Path.Combine(dir.Trim(), "yt-dlp.exe");
                    if (File.Exists(p)) return p;
                }
                catch { }
            }
        }

        return null;
    }

    /// <summary>
    /// Automatically downloads the official standalone yt-dlp.exe into %LOCALAPPDATA%\SoundPulse\
    /// on first YouTube/SoundCloud/etc. link so the user never has to install anything manually.
    /// </summary>
    public async Task<string> EnsureYtDlpAsync(IProgress<double>? progress = null, CancellationToken ct = default)
    {
        string? existing = FindYtDlpExecutable();
        if (existing != null)
            return existing;

        string targetPath = ManagedYtDlpPath;
        string partPath = targetPath + ".part";

        try
        {
            using var response = await _http.GetAsync(YtDlpDownloadUrl, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength;
            await using var netStream = await response.Content.ReadAsStreamAsync(ct);
            await using (var fs = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true))
            {
                var buffer = new byte[65536];
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fs.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes is > 0)
                        progress?.Report(Math.Clamp(100.0 * downloaded / totalBytes.Value, 0, 99));
                }
            }

            if (File.Exists(targetPath)) File.Delete(targetPath);
            File.Move(partPath, targetPath);
            progress?.Report(100.0);
            return targetPath;
        }
        catch
        {
            try { if (File.Exists(partPath)) File.Delete(partPath); } catch { }
            throw;
        }
    }

    /// <summary>
    /// Searches the open Audius catalog (millions of free full-length MP3 tracks, no API key).
    /// </summary>
    public async Task<List<SearchResultItem>> SearchAudiusAsync(string query, int limit = 15, CancellationToken ct = default)
    {
        string url = $"{AudiusApiBase}/tracks/search?query={Uri.EscapeDataString(query)}&app_name={AudiusAppName}";
        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();

        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var results = new List<SearchResultItem>();
        if (!doc.RootElement.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Array)
            return results;

        foreach (var track in data.EnumerateArray())
        {
            string? id = track.TryGetProperty("id", out var idEl) ? idEl.GetString() : null;
            string? title = track.TryGetProperty("title", out var titleEl) ? titleEl.GetString() : null;
            if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(title))
                continue;

            string artist = "Неизвестный исполнитель";
            if (track.TryGetProperty("user", out var userEl) && userEl.ValueKind == JsonValueKind.Object)
            {
                if (userEl.TryGetProperty("name", out var nameEl) && !string.IsNullOrWhiteSpace(nameEl.GetString()))
                    artist = nameEl.GetString()!.Trim();
            }

            TimeSpan? duration = null;
            if (track.TryGetProperty("duration", out var durEl) && durEl.TryGetInt32(out int secs) && secs > 0)
                duration = TimeSpan.FromSeconds(secs);

            string thumb = string.Empty;
            if (track.TryGetProperty("artwork", out var artEl) && artEl.ValueKind == JsonValueKind.Object)
            {
                if (artEl.TryGetProperty("480x480", out var a480) && !string.IsNullOrWhiteSpace(a480.GetString()))
                    thumb = a480.GetString()!;
                else if (artEl.TryGetProperty("150x150", out var a150) && !string.IsNullOrWhiteSpace(a150.GetString()))
                    thumb = a150.GetString()!;
            }

            results.Add(new SearchResultItem
            {
                Id = id,
                Title = title.Trim(),
                Artist = artist,
                StreamUrl = $"{AudiusApiBase}/tracks/{Uri.EscapeDataString(id)}/stream?app_name={AudiusAppName}",
                ThumbnailUrl = thumb,
                SourceLabel = "Audius",
                Duration = duration
            });

            if (results.Count >= limit)
                break;
        }

        return results;
    }

    public async Task<string> DownloadSearchResultAsync(
        SearchResultItem item,
        string destinationFolder,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(destinationFolder);

        string baseName = SanitizeFileName($"{item.Artist} - {item.Title}");
        if (string.IsNullOrWhiteSpace(baseName))
            baseName = item.Id;

        string targetPath = EnsureUniquePath(destinationFolder, baseName, ".mp3");
        string partPath = targetPath + ".part";

        try
        {
            using var response = await _http.GetAsync(item.StreamUrl, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength;
            await using var netStream = await response.Content.ReadAsStreamAsync(ct);
            await using (var fs = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true))
            {
                var buffer = new byte[65536];
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fs.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes is > 0)
                        progress?.Report(Math.Clamp(100.0 * downloaded / totalBytes.Value, 0, 99));
                }
            }

            if (File.Exists(targetPath)) File.Delete(targetPath);
            File.Move(partPath, targetPath);

            await TryEmbedMetadataAsync(targetPath, item.Title, item.Artist, item.ThumbnailUrl, ct);
            progress?.Report(100.0);
            return targetPath;
        }
        catch
        {
            try { if (File.Exists(partPath)) File.Delete(partPath); } catch { }
            throw;
        }
    }

    /// <summary>
    /// Downloads a direct HTTP/HTTPS audio URL (.mp3, .m4a, .flac, .ogg, .wav, etc.).
    /// </summary>
    public async Task<string> DownloadDirectAudioUrlAsync(
        string url,
        string destinationFolder,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(destinationFolder);

        using var response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        string mediaType = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant() ?? string.Empty;
        if (mediaType.StartsWith("text/html", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "Это веб-страница, а не прямой аудиофайл. Для YouTube/SoundCloud положите yt-dlp.exe рядом с SoundPulse.exe.");
        }

        string? fileName = response.Content.Headers.ContentDisposition?.FileNameStar
            ?? response.Content.Headers.ContentDisposition?.FileName?.Trim('"');

        if (string.IsNullOrWhiteSpace(fileName))
        {
            var uri = new Uri(url);
            fileName = Uri.UnescapeDataString(Path.GetFileName(uri.AbsolutePath));
        }

        string ext = Path.GetExtension(fileName ?? string.Empty);
        if (string.IsNullOrWhiteSpace(ext) || !AudioDecoder.SupportedExtensions.Contains(ext))
        {
            ext = mediaType switch
            {
                "audio/mpeg" or "audio/mp3" => ".mp3",
                "audio/mp4" or "audio/aac" or "audio/x-m4a" => ".m4a",
                "audio/ogg" or "application/ogg" => ".ogg",
                "audio/flac" or "audio/x-flac" => ".flac",
                "audio/wav" or "audio/x-wav" => ".wav",
                _ => ".mp3"
            };
        }

        string baseName = SanitizeFileName(Path.GetFileNameWithoutExtension(fileName ?? string.Empty));
        if (string.IsNullOrWhiteSpace(baseName))
            baseName = $"track_{DateTime.Now:yyyyMMdd_HHmmss}";

        string targetPath = EnsureUniquePath(destinationFolder, baseName, ext);
        string partPath = targetPath + ".part";

        try
        {
            long? totalBytes = response.Content.Headers.ContentLength;
            await using var netStream = await response.Content.ReadAsStreamAsync(ct);
            await using (var fs = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true))
            {
                var buffer = new byte[65536];
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fs.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes is > 0)
                        progress?.Report(Math.Clamp(100.0 * downloaded / totalBytes.Value, 0, 99));
                }
            }

            if (File.Exists(targetPath)) File.Delete(targetPath);
            File.Move(partPath, targetPath);
            progress?.Report(100.0);
            return targetPath;
        }
        catch
        {
            try { if (File.Exists(partPath)) File.Delete(partPath); } catch { }
            throw;
        }
    }

    /// <summary>
    /// Downloads audio from YouTube / SoundCloud / Bandcamp / 1000+ sites via yt-dlp.exe
    /// and embeds clean Title, Artist & Cover Art via TagLibSharp (no ffmpeg required).
    /// </summary>
    public async Task<List<string>> DownloadViaYtDlpAsync(
        string ytDlpPath,
        string url,
        string destinationFolder,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(destinationFolder);

        var startUtc = DateTime.UtcNow.AddSeconds(-2);
        var beforeFiles = new HashSet<string>(
            Directory.EnumerateFiles(destinationFolder),
            StringComparer.OrdinalIgnoreCase);

        bool isExplicitPlaylist = url.Contains("/playlist?", StringComparison.OrdinalIgnoreCase);

        string outputTemplate = Path.Combine(destinationFolder, "%(uploader,artist).40s - %(title).80s.%(ext)s");
        var psi = new ProcessStartInfo
        {
            FileName = ytDlpPath,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
            StandardErrorEncoding = System.Text.Encoding.UTF8
        };
        // Force Python/yt-dlp to emit UTF-8 on Windows even for Cyrillic/Unicode file paths
        psi.Environment["PYTHONUTF8"] = "1";
        psi.Environment["PYTHONIOENCODING"] = "utf-8";

        psi.ArgumentList.Add("-f");
        psi.ArgumentList.Add("bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]/bestaudio[ext=mp3]/best[ext=mp4]/bestaudio");
        if (!isExplicitPlaylist)
            psi.ArgumentList.Add("--no-playlist");
        psi.ArgumentList.Add("--windows-filenames");
        psi.ArgumentList.Add("--no-simulate");
        psi.ArgumentList.Add("--newline");
        psi.ArgumentList.Add("--no-warnings");
        psi.ArgumentList.Add("--print");
        psi.ArgumentList.Add("after_move:SP_FILE:%(filepath)s");
        psi.ArgumentList.Add("--print");
        psi.ArgumentList.Add("after_move:SP_ID:%(id)s");
        psi.ArgumentList.Add("--print");
        psi.ArgumentList.Add("after_move:SP_TITLE:%(title)s");
        psi.ArgumentList.Add("--print");
        psi.ArgumentList.Add("after_move:SP_ARTIST:%(artist,uploader)s");
        psi.ArgumentList.Add("--print");
        psi.ArgumentList.Add("after_move:SP_THUMB:%(thumbnail)s");
        psi.ArgumentList.Add("-o");
        psi.ArgumentList.Add(outputTemplate);
        psi.ArgumentList.Add(url);

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException("Не удалось запустить yt-dlp.exe.");

        var errLines = new List<string>();
        var savedEntries = new List<(string FilePath, string VideoId, string Title, string Artist, string Thumb)>();
        string curFile = string.Empty, curId = string.Empty, curTitle = string.Empty, curArtist = string.Empty, curThumb = string.Empty;

        var stdoutTask = Task.Run(async () =>
        {
            while (await proc.StandardOutput.ReadLineAsync(ct) is { } line)
            {
                if (line.StartsWith("SP_FILE:", StringComparison.Ordinal))
                {
                    if (!string.IsNullOrWhiteSpace(curFile))
                        savedEntries.Add((curFile, curId, curTitle, curArtist, curThumb));
                    curFile = line["SP_FILE:".Length..].Trim();
                    curId = curTitle = curArtist = curThumb = string.Empty;
                }
                else if (line.StartsWith("SP_ID:", StringComparison.Ordinal))
                {
                    curId = line["SP_ID:".Length..].Trim();
                }
                else if (line.StartsWith("SP_TITLE:", StringComparison.Ordinal))
                {
                    curTitle = line["SP_TITLE:".Length..].Trim();
                }
                else if (line.StartsWith("SP_ARTIST:", StringComparison.Ordinal))
                {
                    curArtist = line["SP_ARTIST:".Length..].Trim();
                }
                else if (line.StartsWith("SP_THUMB:", StringComparison.Ordinal))
                {
                    curThumb = line["SP_THUMB:".Length..].Trim();
                }
                else
                {
                    var m = YtDlpPercentRegex.Match(line);
                    if (m.Success && double.TryParse(m.Groups[1].Value, CultureInfo.InvariantCulture, out double pct))
                    {
                        progress?.Report(Math.Clamp(pct, 0, 99));
                    }
                }
            }
            if (!string.IsNullOrWhiteSpace(curFile))
                savedEntries.Add((curFile, curId, curTitle, curArtist, curThumb));
        }, ct);

        var stderrTask = Task.Run(async () =>
        {
            while (await proc.StandardError.ReadLineAsync(ct) is { } line)
            {
                if (!string.IsNullOrWhiteSpace(line))
                    errLines.Add(line);
            }
        }, ct);

        await proc.WaitForExitAsync(ct);
        await Task.WhenAll(stdoutTask, stderrTask);

        if (proc.ExitCode != 0 && savedEntries.Count == 0)
        {
            string details = errLines.LastOrDefault() ?? $"код выхода {proc.ExitCode}";
            throw new InvalidOperationException($"yt-dlp: {details}");
        }

        // Embed clean Title, Artist & JPEG Cover Art (WPF doesn't decode WebP natively!)
        var finalPaths = new List<string>();
        foreach (var entry in savedEntries)
        {
            if (File.Exists(entry.FilePath))
            {
                var (artist, title) = ParseArtistAndTitle(entry.Title, entry.Artist);
                string jpegThumb = ResolveJpegThumbnailUrl(entry.Thumb, entry.VideoId);
                await TryEmbedMetadataAsync(entry.FilePath, title, artist, jpegThumb, ct);
                finalPaths.Add(entry.FilePath);
            }
        }

        if (finalPaths.Count == 0)
        {
            finalPaths = Directory.EnumerateFiles(destinationFolder)
                .Where(f => AudioDecoder.IsSupportedAudioFile(f) &&
                            (!beforeFiles.Contains(f) || File.GetLastWriteTimeUtc(f) >= startUtc))
                .OrderByDescending(File.GetLastWriteTimeUtc)
                .ToList();
        }

        progress?.Report(100.0);
        return finalPaths;
    }

    private static string ResolveJpegThumbnailUrl(string rawThumb, string videoId)
    {
        // YouTube serves WebP thumbnails by default (vi_webp/...webp), which WPF BitmapImage cannot decode natively.
        // Prefer standard JPEG (hqdefault.jpg) for YouTube videos so album covers always render crisply!
        if (!string.IsNullOrWhiteSpace(videoId) &&
            (rawThumb.Contains("ytimg.com", StringComparison.OrdinalIgnoreCase) ||
             rawThumb.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) ||
             string.IsNullOrWhiteSpace(rawThumb)))
        {
            return $"https://i.ytimg.com/vi/{videoId}/hqdefault.jpg";
        }
        return rawThumb;
    }

    private static (string Artist, string Title) ParseArtistAndTitle(string rawTitle, string rawArtist)
    {
        string cleanArtist = rawArtist.Trim();
        if (cleanArtist.EndsWith(" - Topic", StringComparison.OrdinalIgnoreCase))
            cleanArtist = cleanArtist[..^" - Topic".Length].Trim();
        if (cleanArtist.EndsWith("VEVO", StringComparison.Ordinal) && cleanArtist.Length > 4)
            cleanArtist = cleanArtist[..^4].Trim();

        string cleaned = NoiseSuffixRegex.Replace(rawTitle, string.Empty).Trim();
        cleaned = Regex.Replace(cleaned, @"\s{2,}", " ");

        foreach (string sep in new[] { " - ", " – ", " — ", " | " })
        {
            int idx = cleaned.IndexOf(sep, StringComparison.Ordinal);
            if (idx > 0 && idx + sep.Length < cleaned.Length)
            {
                string left = cleaned[..idx].Trim();
                string right = cleaned[(idx + sep.Length)..].Trim();
                if (left.Length >= 2 && right.Length >= 1)
                    return (left, right);
            }
        }

        return (string.IsNullOrWhiteSpace(cleanArtist) ? "Неизвестный исполнитель" : cleanArtist,
                string.IsNullOrWhiteSpace(cleaned) ? rawTitle : cleaned);
    }

    private async Task TryEmbedMetadataAsync(string filePath, string title, string artist, string thumbnailUrl, CancellationToken ct)
    {
        try
        {
            byte[]? coverBytes = null;
            if (!string.IsNullOrWhiteSpace(thumbnailUrl))
            {
                try { coverBytes = await _http.GetByteArrayAsync(thumbnailUrl, ct); }
                catch { }
            }

            using var tagFile = TagLib.File.Create(filePath);
            tagFile.Tag.Title = title;
            tagFile.Tag.Performers = [artist];
            if (coverBytes is { Length: > 0 })
            {
                tagFile.Tag.Pictures =
                [
                    new TagLib.Picture(new TagLib.ByteVector(coverBytes))
                    {
                        Type = TagLib.PictureType.FrontCover,
                        MimeType = "image/jpeg"
                    }
                ];
            }
            tagFile.Save();
        }
        catch
        {
            // Non-fatal
        }
    }

    private static string SanitizeFileName(string name)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, ' ');
        name = Regex.Replace(name, @"\s{2,}", " ").Trim().TrimEnd('.');
        if (name.Length > 110) name = name[..110].Trim();
        return name;
    }

    private static string EnsureUniquePath(string folder, string baseName, string ext)
    {
        string candidate = Path.Combine(folder, baseName + ext);
        if (!File.Exists(candidate)) return candidate;

        for (int i = 2; i < 1000; i++)
        {
            candidate = Path.Combine(folder, $"{baseName} ({i}){ext}");
            if (!File.Exists(candidate)) return candidate;
        }
        return Path.Combine(folder, $"{baseName}_{Guid.NewGuid():N}{ext}");
    }

    public void Dispose()
    {
        _http.Dispose();
    }
}
