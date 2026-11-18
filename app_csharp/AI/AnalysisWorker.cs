using System.Threading.Channels;
using SoundPulse.Domain;

namespace SoundPulse.AI;

/// <summary>
/// Event-driven background analysis queue (0% idle CPU when empty).
/// </summary>
public sealed class AnalysisWorker : IDisposable
{
    private readonly MusicClassifier _classifier;
    private readonly Channel<TrackItem> _channel;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _workerTask;
    private readonly Action<TrackItem, TrackStatus, AnalysisResult?, string?> _onUpdate;
    private bool _disposed;

    public AnalysisWorker(
        MusicClassifier classifier,
        Action<TrackItem, TrackStatus, AnalysisResult?, string?> onUpdate)
    {
        _classifier = classifier;
        _onUpdate = onUpdate;
        _channel = Channel.CreateUnbounded<TrackItem>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false
        });
        _workerTask = Task.Run(ProcessLoopAsync);
    }

    public void Enqueue(TrackItem track)
    {
        if (_disposed) return;
        _channel.Writer.TryWrite(track);
    }

    private async Task ProcessLoopAsync()
    {
        try
        {
            await foreach (var track in _channel.Reader.ReadAllAsync(_cts.Token))
            {
                if (_cts.Token.IsCancellationRequested) break;

                _onUpdate(track, TrackStatus.Analyzing, null, null);

                try
                {
                    var result = _classifier.AnalyzeFile(track.FilePath);
                    _onUpdate(track, TrackStatus.Ready, result, null);
                }
                catch (Exception ex)
                {
                    _onUpdate(track, TrackStatus.Error, null, ex.Message);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Graceful shutdown
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _channel.Writer.TryComplete();
        _cts.Cancel();
        try { _workerTask.Wait(TimeSpan.FromSeconds(1)); } catch { }
        _cts.Dispose();
    }
}
