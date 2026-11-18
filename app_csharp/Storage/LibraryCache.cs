using System.IO;
using System.Text.Json;
using SoundPulse.Domain;

namespace SoundPulse.Storage;

public sealed class CachedTrackRecord
{
    public int AnalysisVersion { get; set; }
    public string FilePath { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public long LastWriteUtcTicks { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Artist { get; set; } = string.Empty;
    public string Album { get; set; } = string.Empty;
    public double DurationSec { get; set; }
    public string Genre { get; set; } = string.Empty;
    public float GenreConfidence { get; set; }
    public string Vibe { get; set; } = string.Empty;
    public int VibeIndex { get; set; } = -1;
    public float VibeConfidence { get; set; }
    public int Bpm { get; set; }
    public float Energy { get; set; }
    public List<ScoreEntry> TopGenres { get; set; } = [];
    public List<ScoreEntry> VibeScores { get; set; } = [];
}

public static class LibraryCache
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public static string CacheFilePath
    {
        get
        {
            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SoundPulse");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "library.json");
        }
    }

    public static List<CachedTrackRecord> Load()
    {
        try
        {
            string path = CacheFilePath;
            if (!File.Exists(path)) return [];

            string json = File.ReadAllText(path);
            var records = JsonSerializer.Deserialize<List<CachedTrackRecord>>(json);
            return records ?? [];
        }
        catch
        {
            return [];
        }
    }

    public const int CurrentAnalysisVersion = 3; // Hybrid 3-crop ONNX + DSP BPM/Energy fusion

    public static int ComputeCacheVersion(ModelLabels? labels)
        => CurrentAnalysisVersion + (labels?.Version ?? 1) * 100 + (labels?.Vibes.Count ?? 3) * 10;

    public static void Save(IEnumerable<TrackItem> tracks, int analysisVersion = CurrentAnalysisVersion)
    {
        try
        {
            var records = tracks
                .Where(t => File.Exists(t.FilePath))
                .Select(t => new CachedTrackRecord
                {
                    AnalysisVersion = analysisVersion,
                    FilePath = t.FilePath,
                    FileSize = t.FileSize,
                    LastWriteUtcTicks = t.LastWriteUtcTicks,
                    Title = t.Title,
                    Artist = t.Artist,
                    Album = t.Album,
                    DurationSec = t.DurationSec,
                    Genre = t.Genre,
                    GenreConfidence = t.GenreConfidence,
                    Vibe = t.Vibe,
                    VibeIndex = t.VibeIndex,
                    VibeConfidence = t.VibeConfidence,
                    Bpm = t.Bpm,
                    Energy = t.Energy,
                    TopGenres = t.TopGenres,
                    VibeScores = t.VibeScores
                })
                .ToList();

            string path = CacheFilePath;
            string tmpPath = path + ".tmp";
            File.WriteAllText(tmpPath, JsonSerializer.Serialize(records, JsonOptions));
            File.Move(tmpPath, path, overwrite: true);
        }
        catch
        {
            // Non-fatal: cache write errors must never crash playback
        }
    }
}
