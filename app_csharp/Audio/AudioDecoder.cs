using System.IO;
using NAudio.Vorbis;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace SoundPulse.Audio;

public sealed record TrackMetadata(
    string Title,
    string Artist,
    string Album,
    double DurationSec);

public static class AudioDecoder
{
    public static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".mp3", ".wav", ".flac", ".m4a", ".mp4", ".aac", ".wma", ".ogg", ".opus", ".webm", ".aiff", ".aif"
    };

    public static bool IsSupportedAudioFile(string path)
        => !string.IsNullOrWhiteSpace(path) && SupportedExtensions.Contains(Path.GetExtension(path));

    public static WaveStream OpenWaveStream(string filePath)
    {
        bool isOggHeader = HasMagicHeader(filePath, "OggS"u8);

        if (isOggHeader)
        {
            try
            {
                return new VorbisWaveReader(filePath);
            }
            catch
            {
                // Fall back to MediaFoundation (e.g. Opus in Ogg)
            }
        }

        try
        {
            return new AudioFileReader(filePath);
        }
        catch
        {
            // MediaFoundation sniffs container bytes even if file extension is misnamed (e.g. M4A named .ogg)
            return new MediaFoundationReader(filePath);
        }
    }

    private static bool HasMagicHeader(string filePath, ReadOnlySpan<byte> magic)
    {
        try
        {
            Span<byte> header = stackalloc byte[magic.Length];
            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            int read = fs.Read(header);
            return read == magic.Length && header.SequenceEqual(magic);
        }
        catch
        {
            return false;
        }
    }

    public static ISampleProvider ToSampleProvider(WaveStream stream)
    {
        if (stream is ISampleProvider sp)
            return sp;
        return stream.ToSampleProvider();
    }

    /// <summary>
    /// Decodes up to 3 representative 10-second slices (25%, 50%, 75%) into 16 kHz mono PCM
    /// for stable multi-crop TTA inference (avoids bridge/intro misfires).
    /// </summary>
    public static (List<float[]> Chunks, double DurationSec) DecodeRepresentativeChunks16kMono(
        string filePath,
        int targetSampleRate = 16000,
        double chunkDurationSec = 10.0)
    {
        int targetSamples = (int)(targetSampleRate * chunkDurationSec);

        using var stream = OpenWaveStream(filePath);
        double durationSec = stream.TotalTime.TotalSeconds;

        // Choose slice start positions: 3 crops for tracks >= 22s, 1 middle crop for shorter tracks
        double[] startPositions;
        if (stream.CanSeek && durationSec >= chunkDurationSec * 2.2)
        {
            double maxStart = Math.Max(0.0, durationSec - chunkDurationSec);
            startPositions =
            [
                Math.Clamp(durationSec * 0.25 - chunkDurationSec * 0.5, 0.0, maxStart),
                Math.Clamp(durationSec * 0.50 - chunkDurationSec * 0.5, 0.0, maxStart),
                Math.Clamp(durationSec * 0.75 - chunkDurationSec * 0.5, 0.0, maxStart)
            ];
        }
        else if (stream.CanSeek && durationSec > chunkDurationSec + 0.5)
        {
            startPositions = [(durationSec - chunkDurationSec) / 2.0];
        }
        else
        {
            startPositions = [0.0];
        }

        var chunks = new List<float[]>(startPositions.Length);
        var readBuffer = new float[16384];

        foreach (double startSec in startPositions)
        {
            if (stream.CanSeek)
            {
                try { stream.CurrentTime = TimeSpan.FromSeconds(startSec); }
                catch { stream.Position = 0; }
            }

            ISampleProvider provider = ToSampleProvider(stream);
            if (provider.WaveFormat.Channels > 1)
                provider = new MultiChannelToMonoProvider(provider);
            if (provider.WaveFormat.SampleRate != targetSampleRate)
                provider = new WdlResamplingSampleProvider(provider, targetSampleRate);

            var chunk = new float[targetSamples];
            int totalRead = 0;
            while (totalRead < targetSamples)
            {
                int toRead = Math.Min(readBuffer.Length, targetSamples - totalRead);
                int read = provider.Read(readBuffer, 0, toRead);
                if (read <= 0) break;

                Array.Copy(readBuffer, 0, chunk, totalRead, read);
                totalRead += read;
            }

            if (durationSec <= 0 && totalRead > 0)
                durationSec = (double)totalRead / targetSampleRate;

            // Keep chunk if it contains audio (or if it's the only chunk)
            if (totalRead > 0 || chunks.Count == 0)
                chunks.Add(chunk);
        }

        return (chunks, durationSec);
    }

    public static TrackMetadata ReadMetadata(string filePath)
    {
        string fallbackTitle = Path.GetFileNameWithoutExtension(filePath);
        string fallbackArtist = "Неизвестный исполнитель";

        // Split "Artist - Title" if filename follows common convention
        int sepIdx = fallbackTitle.IndexOf(" - ", StringComparison.Ordinal);
        if (sepIdx > 0 && sepIdx + 3 < fallbackTitle.Length)
        {
            fallbackArtist = fallbackTitle[..sepIdx].Trim();
            fallbackTitle = fallbackTitle[(sepIdx + 3)..].Trim();
        }

        try
        {
            using var tagFile = TagLib.File.Create(filePath);
            string title = !string.IsNullOrWhiteSpace(tagFile.Tag.Title)
                ? tagFile.Tag.Title.Trim()
                : fallbackTitle;
            string artist = !string.IsNullOrWhiteSpace(tagFile.Tag.FirstPerformer)
                ? tagFile.Tag.FirstPerformer.Trim()
                : (!string.IsNullOrWhiteSpace(tagFile.Tag.FirstAlbumArtist)
                    ? tagFile.Tag.FirstAlbumArtist.Trim()
                    : fallbackArtist);
            string album = tagFile.Tag.Album?.Trim() ?? string.Empty;
            double durationSec = tagFile.Properties?.Duration.TotalSeconds ?? 0.0;

            return new TrackMetadata(title, artist, album, durationSec);
        }
        catch
        {
            return new TrackMetadata(fallbackTitle, fallbackArtist, string.Empty, 0.0);
        }
    }

    public static byte[]? ReadCoverArtBytes(string filePath)
    {
        try
        {
            using var tagFile = TagLib.File.Create(filePath);
            var pic = tagFile.Tag.Pictures?.FirstOrDefault();
            if (pic?.Data?.Data is { Length: > 0 } data)
            {
                return data;
            }
        }
        catch
        {
            // Ignore malformed tags
        }
        return null;
    }

    private sealed class MultiChannelToMonoProvider : ISampleProvider
    {
        private readonly ISampleProvider _source;
        private readonly int _channels;
        private float[] _sourceBuffer = [];

        public WaveFormat WaveFormat { get; }

        public MultiChannelToMonoProvider(ISampleProvider source)
        {
            _source = source;
            _channels = source.WaveFormat.Channels;
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(source.WaveFormat.SampleRate, 1);
        }

        public int Read(float[] buffer, int offset, int count)
        {
            int needed = count * _channels;
            if (_sourceBuffer.Length < needed)
                _sourceBuffer = new float[needed];

            int read = _source.Read(_sourceBuffer, 0, needed);
            int frames = read / _channels;
            float invChannels = 1f / _channels;

            for (int f = 0; f < frames; f++)
            {
                float sum = 0f;
                int baseIdx = f * _channels;
                for (int c = 0; c < _channels; c++)
                    sum += _sourceBuffer[baseIdx + c];
                buffer[offset + f] = sum * invChannels;
            }

            return frames;
        }
    }
}
