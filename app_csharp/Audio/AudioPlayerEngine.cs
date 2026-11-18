using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace SoundPulse.Audio;

public sealed class AudioPlayerEngine : IDisposable
{
    private readonly object _sync = new();
    private WaveOutEvent? _outputDevice;
    private WaveStream? _waveStream;
    private VolumeSampleProvider? _volumeProvider;
    private SpectrumMeterProvider? _meterProvider;

    private float _volume = 0.8f;
    private bool _isMuted;
    private long _playGeneration;
    private bool _suppressNextStop;
    private TimeSpan _lastSeekPosition;
    private long _seekHoldUntilTick;
    private bool _disposed;

    public event Action? TrackEnded;
    public event Action? PlaybackStateChanged;

    public bool HasTrack
    {
        get { lock (_sync) return _waveStream != null; }
    }

    public bool IsPlaying
    {
        get { lock (_sync) return _outputDevice?.PlaybackState == PlaybackState.Playing; }
    }

    public bool IsPaused
    {
        get { lock (_sync) return _outputDevice?.PlaybackState == PlaybackState.Paused; }
    }

    public TimeSpan Position
    {
        get
        {
            lock (_sync)
            {
                if (_waveStream == null) return TimeSpan.Zero;
                if (Environment.TickCount64 < _seekHoldUntilTick)
                    return _lastSeekPosition;
                try { return _waveStream.CurrentTime; }
                catch { return TimeSpan.Zero; }
            }
        }
    }

    public TimeSpan Duration
    {
        get
        {
            lock (_sync)
            {
                if (_waveStream == null) return TimeSpan.Zero;
                try { return _waveStream.TotalTime; }
                catch { return TimeSpan.Zero; }
            }
        }
    }

    public float Volume
    {
        get { lock (_sync) return _volume; }
        set
        {
            lock (_sync)
            {
                _volume = Math.Clamp(value, 0f, 1f);
                ApplyVolumeLocked();
            }
        }
    }

    public bool IsMuted
    {
        get { lock (_sync) return _isMuted; }
        set
        {
            lock (_sync)
            {
                _isMuted = value;
                ApplyVolumeLocked();
            }
        }
    }

    public void LoadAndPlay(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        long generation;
        lock (_sync)
        {
            generation = ++_playGeneration;
            _suppressNextStop = false;
            _seekHoldUntilTick = 0;
            DisposePlaybackLocked();

            var stream = AudioDecoder.OpenWaveStream(filePath);
            var sampleProvider = AudioDecoder.ToSampleProvider(stream);
            var volumeProvider = new VolumeSampleProvider(sampleProvider)
            {
                Volume = _isMuted ? 0f : _volume
            };
            var meterProvider = new SpectrumMeterProvider(volumeProvider);

            var output = new WaveOutEvent
            {
                DesiredLatency = 120
            };

            long capturedGen = generation;
            output.PlaybackStopped += (_, stoppedArgs) =>
            {
                bool naturalEnd = false;
                lock (_sync)
                {
                    if (_suppressNextStop)
                    {
                        _suppressNextStop = false;
                        return;
                    }

                    // Manual Stop()/LoadAndPlay() always increments _playGeneration first,
                    // and Pause() never fires PlaybackStopped. So matching generation + Stopped
                    // reliably indicates true EOF even on M4A/VBR streams with inaccurate TotalTime.
                    if (capturedGen == _playGeneration &&
                        _waveStream != null &&
                        stoppedArgs.Exception == null &&
                        _outputDevice?.PlaybackState == PlaybackState.Stopped)
                    {
                        naturalEnd = true;
                    }
                }

                PlaybackStateChanged?.Invoke();
                if (naturalEnd)
                {
                    TrackEnded?.Invoke();
                }
            };

            output.Init(meterProvider);
            output.Play();

            _waveStream = stream;
            _volumeProvider = volumeProvider;
            _meterProvider = meterProvider;
            _outputDevice = output;
        }

        PlaybackStateChanged?.Invoke();
    }

    public void TogglePlayPause()
    {
        lock (_sync)
        {
            if (_outputDevice == null) return;
            if (_outputDevice.PlaybackState == PlaybackState.Playing)
            {
                _outputDevice.Pause();
            }
            else
            {
                _outputDevice.Play();
            }
        }
        PlaybackStateChanged?.Invoke();
    }

    public void Pause()
    {
        lock (_sync)
        {
            if (_outputDevice?.PlaybackState == PlaybackState.Playing)
                _outputDevice.Pause();
        }
        PlaybackStateChanged?.Invoke();
    }

    public void Resume()
    {
        lock (_sync)
        {
            if (_outputDevice?.PlaybackState == PlaybackState.Paused)
                _outputDevice.Play();
        }
        PlaybackStateChanged?.Invoke();
    }

    public void Stop()
    {
        lock (_sync)
        {
            _playGeneration++;
            DisposePlaybackLocked();
        }
        PlaybackStateChanged?.Invoke();
    }

    public void Seek(TimeSpan position)
    {
        lock (_sync)
        {
            if (_waveStream == null || !_waveStream.CanSeek) return;
            var clamped = TimeSpan.FromSeconds(Math.Clamp(position.TotalSeconds, 0, Math.Max(0, _waveStream.TotalTime.TotalSeconds - 0.15)));
            bool wasPlaying = _outputDevice?.PlaybackState == PlaybackState.Playing;

            try
            {
                if (wasPlaying && _outputDevice != null)
                {
                    _outputDevice.Pause();
                }

                _waveStream.CurrentTime = clamped;
                _lastSeekPosition = clamped;
                _seekHoldUntilTick = Environment.TickCount64 + 450;

                if (wasPlaying && _outputDevice != null)
                {
                    _outputDevice.Play();
                }
            }
            catch
            {
                if (wasPlaying && _outputDevice != null)
                {
                    try { _outputDevice.Play(); } catch { }
                }
            }
        }
    }

    public void CopyMeterLevels(float[] destination)
    {
        SpectrumMeterProvider? meter;
        bool playing;
        lock (_sync)
        {
            meter = _meterProvider;
            playing = _outputDevice?.PlaybackState == PlaybackState.Playing;
        }

        if (!playing || meter == null)
        {
            Array.Clear(destination);
            return;
        }

        meter.CopyLevels(destination);
    }

    private void ApplyVolumeLocked()
    {
        if (_volumeProvider != null)
        {
            _volumeProvider.Volume = _isMuted ? 0f : _volume;
        }
    }

    private void DisposePlaybackLocked()
    {
        if (_outputDevice != null)
        {
            try { _outputDevice.Stop(); } catch { }
            _outputDevice.Dispose();
            _outputDevice = null;
        }

        if (_waveStream != null)
        {
            _waveStream.Dispose();
            _waveStream = null;
        }

        _volumeProvider = null;
        _meterProvider = null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_sync)
        {
            _playGeneration++;
            DisposePlaybackLocked();
        }
    }

    /// <summary>
    /// Real 12-band logarithmic FFT spectrum analyzer.
    /// Audio thread only copies samples into a lock-free ring buffer (zero allocations).
    /// FFT runs on demand only during active playback when the UI timer polls CopyLevels().
    /// </summary>
    private sealed class SpectrumMeterProvider : ISampleProvider
    {
        private const int FftSize = 512;
        private const int BarCount = 12;

        private readonly ISampleProvider _source;
        private readonly int _channels;
        private readonly int _sampleRate;
        private readonly float[] _ring = new float[FftSize];
        private int _ringWritePos;

        private readonly float[] _window = new float[FftSize];
        private readonly int[] _bitReverse = new int[FftSize];
        private readonly float[] _cosTable = new float[FftSize / 2];
        private readonly float[] _sinTable = new float[FftSize / 2];
        private readonly float[] _real = new float[FftSize];
        private readonly float[] _imag = new float[FftSize];
        private readonly (int StartBin, int EndBin)[] _bandRanges = new (int, int)[BarCount];
        private readonly float[] _smoothed = new float[BarCount];

        public WaveFormat WaveFormat => _source.WaveFormat;

        public SpectrumMeterProvider(ISampleProvider source)
        {
            _source = source;
            _channels = Math.Max(1, source.WaveFormat.Channels);
            _sampleRate = Math.Max(8000, source.WaveFormat.SampleRate);

            int bits = (int)Math.Log2(FftSize);
            for (int i = 0; i < FftSize; i++)
            {
                _window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / FftSize));
                int rev = 0, x = i;
                for (int b = 0; b < bits; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
                _bitReverse[i] = rev;
            }

            for (int i = 0; i < FftSize / 2; i++)
            {
                double angle = -2.0 * Math.PI * i / FftSize;
                _cosTable[i] = (float)Math.Cos(angle);
                _sinTable[i] = (float)Math.Sin(angle);
            }

            // 12 logarithmic frequency bands from ~55 Hz (sub-bass) to ~13.5 kHz (treble)
            double[] edgesHz = [55, 95, 155, 250, 400, 650, 1050, 1700, 2700, 4300, 6800, 10200, 14500];
            double binHz = (double)_sampleRate / FftSize;
            int maxBin = FftSize / 2 - 1;

            for (int b = 0; b < BarCount; b++)
            {
                int start = Math.Clamp((int)Math.Floor(edgesHz[b] / binHz), 1, maxBin);
                int end = Math.Clamp((int)Math.Ceiling(edgesHz[b + 1] / binHz), start, maxBin);
                _bandRanges[b] = (start, end);
            }
        }

        public int Read(float[] buffer, int offset, int count)
        {
            int read = _source.Read(buffer, offset, count);
            if (read <= 0) return read;

            int pos = _ringWritePos;
            int step = _channels;
            for (int i = offset; i + step <= offset + read; i += step)
            {
                float mono = step == 1 ? buffer[i] : 0.5f * (buffer[i] + buffer[i + 1]);
                _ring[pos] = mono;
                pos = (pos + 1) & (FftSize - 1);
            }
            Volatile.Write(ref _ringWritePos, pos);

            return read;
        }

        public void CopyLevels(float[] destination)
        {
            int readStart = Volatile.Read(ref _ringWritePos);

            // Copy ring buffer into bit-reversed windowed FFT input
            for (int i = 0; i < FftSize; i++)
            {
                float s = _ring[(readStart + i) & (FftSize - 1)];
                int rev = _bitReverse[i];
                _real[rev] = s * _window[i];
                _imag[rev] = 0f;
            }

            // In-place 512-point Radix-2 FFT
            for (int size = 2; size <= FftSize; size <<= 1)
            {
                int half = size >> 1;
                int tableStep = FftSize / size;
                for (int i = 0; i < FftSize; i += size)
                {
                    int k = 0;
                    for (int j = i; j < i + half; j++)
                    {
                        int p = j + half;
                        float c = _cosTable[k];
                        float s = _sinTable[k];
                        float tr = _real[p] * c - _imag[p] * s;
                        float ti = _real[p] * s + _imag[p] * c;
                        _real[p] = _real[j] - tr;
                        _imag[p] = _imag[j] - ti;
                        _real[j] += tr;
                        _imag[j] += ti;
                        k += tableStep;
                    }
                }
            }

            // Compute perceptual dB magnitude per frequency band with tilt compensation
            int n = Math.Min(destination.Length, BarCount);
            const float invNorm = 2f / FftSize;

            for (int b = 0; b < n; b++)
            {
                var (startBin, endBin) = _bandRanges[b];
                float maxMag = 0f;
                for (int bin = startBin; bin <= endBin; bin++)
                {
                    float mag = MathF.Sqrt(_real[bin] * _real[bin] + _imag[bin] * _imag[bin]) * invNorm;
                    if (mag > maxMag) maxMag = mag;
                }

                // Gentle high-frequency tilt (+1.2 dB per band) so treble dances alongside bass
                float db = 20f * MathF.Log10(MathF.Max(maxMag, 1e-5f)) + b * 1.15f;
                float norm = Math.Clamp((db + 54f) / 46f, 0f, 1f);
                norm = MathF.Pow(norm, 1.15f);

                // Fast attack, smooth gravity decay
                float prev = _smoothed[b];
                float next = norm > prev
                    ? prev * 0.30f + norm * 0.70f
                    : MathF.Max(0f, prev * 0.78f - 0.015f);

                _smoothed[b] = next;
                destination[b] = next;
            }
        }
    }
}
