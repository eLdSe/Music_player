using System.Collections.Concurrent;
using System.IO;
using SoundPulse.Audio;

namespace SoundPulse.Services;

/// <summary>
/// Event-driven Watch Folder service using OS FileSystemWatcher (0% CPU when idle).
/// Automatically detects newly saved/downloaded audio files (e.g. from browser or Telegram)
/// and waits until the file write completes before firing AudioFileArrived.
/// </summary>
public sealed class WatchFolderService : IDisposable
{
    private readonly Action<string> _onAudioFileArrived;
    private readonly ConcurrentDictionary<string, byte> _inFlight = new(StringComparer.OrdinalIgnoreCase);
    private FileSystemWatcher? _watcher;
    private CancellationTokenSource _cts = new();
    private bool _disposed;

    public string? ActivePath { get; private set; }
    public bool IsWatching => _watcher?.EnableRaisingEvents == true;

    public WatchFolderService(Action<string> onAudioFileArrived)
    {
        _onAudioFileArrived = onAudioFileArrived;
    }

    public bool Configure(bool enabled, string? folderPath)
    {
        Stop();

        if (!enabled || string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            return false;

        try
        {
            _cts = new CancellationTokenSource();
            var watcher = new FileSystemWatcher(folderPath)
            {
                IncludeSubdirectories = false,
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.Size | NotifyFilters.LastWrite,
                InternalBufferSize = 32768
            };

            watcher.Created += OnFileEvent;
            watcher.Renamed += OnRenamedEvent;
            watcher.EnableRaisingEvents = true;

            _watcher = watcher;
            ActivePath = folderPath;
            return true;
        }
        catch
        {
            Stop();
            return false;
        }
    }

    public void Stop()
    {
        if (_watcher != null)
        {
            _watcher.EnableRaisingEvents = false;
            _watcher.Created -= OnFileEvent;
            _watcher.Renamed -= OnRenamedEvent;
            _watcher.Dispose();
            _watcher = null;
        }

        _cts.Cancel();
        _cts.Dispose();
        _cts = new CancellationTokenSource();
        _inFlight.Clear();
        ActivePath = null;
    }

    private void OnFileEvent(object sender, FileSystemEventArgs e)
        => ScheduleIfAudio(e.FullPath);

    private void OnRenamedEvent(object sender, RenamedEventArgs e)
        => ScheduleIfAudio(e.FullPath);

    private void ScheduleIfAudio(string fullPath)
    {
        if (_disposed || !AudioDecoder.IsSupportedAudioFile(fullPath))
            return;

        if (!_inFlight.TryAdd(fullPath, 0))
            return;

        var token = _cts.Token;
        _ = Task.Run(async () =>
        {
            try
            {
                if (await WaitUntilFileReadyAsync(fullPath, token))
                {
                    _onAudioFileArrived(fullPath);
                }
            }
            catch (OperationCanceledException)
            {
                // Stopped
            }
            finally
            {
                _inFlight.TryRemove(fullPath, out _);
            }
        }, token);
    }

    private static async Task<bool> WaitUntilFileReadyAsync(string path, CancellationToken ct)
    {
        // Wait briefly for browser/Telegram to finish writing & closing the file
        long lastLen = -1;
        for (int attempt = 0; attempt < 25; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Delay(350, ct);

            if (!File.Exists(path)) return false;

            try
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                long len = fs.Length;
                if (len > 4096 && len == lastLen)
                    return true;
                lastLen = len;
            }
            catch (IOException)
            {
                // Still being written
            }
            catch (UnauthorizedAccessException)
            {
                // Still locked
            }
        }

        return File.Exists(path);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Stop();
    }
}
