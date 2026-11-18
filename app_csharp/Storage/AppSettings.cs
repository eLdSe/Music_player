using System.IO;
using System.Text.Json;

namespace SoundPulse.Storage;

public sealed class AppSettings
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public string DownloadFolder { get; set; } = GetDefaultDownloadFolder();
    public bool AutoSortByCategory { get; set; } = false;
    public bool WatchFolderEnabled { get; set; } = false;
    public string WatchFolderPath { get; set; } = string.Empty;

    public static string GetDefaultDownloadFolder()
    {
        string music = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
        if (string.IsNullOrWhiteSpace(music))
            music = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(music, "SoundPulse");
    }

    public static string SettingsFilePath
    {
        get
        {
            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SoundPulse");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "settings.json");
        }
    }

    public static AppSettings Load()
    {
        try
        {
            string path = SettingsFilePath;
            if (!File.Exists(path))
                return new AppSettings();

            var loaded = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(path));
            if (loaded == null)
                return new AppSettings();

            if (string.IsNullOrWhiteSpace(loaded.DownloadFolder))
                loaded.DownloadFolder = GetDefaultDownloadFolder();

            return loaded;
        }
        catch
        {
            return new AppSettings();
        }
    }

    public void Save()
    {
        try
        {
            string path = SettingsFilePath;
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, JsonSerializer.Serialize(this, JsonOptions));
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            // Non-fatal
        }
    }
}
