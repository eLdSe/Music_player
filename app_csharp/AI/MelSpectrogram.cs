namespace SoundPulse.AI;

/// <summary>
/// Bit-accurate pure C# port of torchaudio.transforms.MelSpectrogram + AmplitudeToDB
/// matching ml/dataset.py and ml/test_inference.py:
///   sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512,
///   n_mels=128, power=2.0, center=True (reflect pad), mel_scale="htk", norm=None,
///   AmplitudeToDB(stype="power", amin=1e-10), z-score normalization.
/// </summary>
public sealed class MelSpectrogram
{
    public int SampleRate { get; }
    public int NFft { get; }
    public int HopLength { get; }
    public int NMels { get; }
    public int TargetFrames { get; }

    private readonly float[] _window;
    private readonly int[] _bitReverse;
    private readonly float[] _cosTable;
    private readonly float[] _sinTable;
    private readonly MelBand[] _melBands;

    private readonly struct MelBand
    {
        public readonly int StartBin;
        public readonly float[] Weights;

        public MelBand(int startBin, float[] weights)
        {
            StartBin = startBin;
            Weights = weights;
        }
    }

    public MelSpectrogram(int sampleRate = 16000, int nFft = 1024, int hopLength = 512, int nMels = 128, int chunkSamples = 160000)
    {
        if ((nFft & (nFft - 1)) != 0)
            throw new ArgumentException("nFft must be a power of 2.", nameof(nFft));

        SampleRate = sampleRate;
        NFft = nFft;
        HopLength = hopLength;
        NMels = nMels;
        // center=True pads by nFft/2 on both sides -> floor(chunkSamples / hopLength) + 1 = 313
        TargetFrames = chunkSamples / hopLength + 1;

        // Periodic Hann window (matches torch.hann_window(nFft, periodic=True))
        _window = new float[nFft];
        for (int i = 0; i < nFft; i++)
        {
            _window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / nFft));
        }

        // Precompute FFT bit-reversal and twiddle factors
        int bits = (int)Math.Log2(nFft);
        _bitReverse = new int[nFft];
        for (int i = 0; i < nFft; i++)
        {
            int rev = 0;
            int x = i;
            for (int b = 0; b < bits; b++)
            {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            _bitReverse[i] = rev;
        }

        _cosTable = new float[nFft / 2];
        _sinTable = new float[nFft / 2];
        for (int i = 0; i < nFft / 2; i++)
        {
            double angle = -2.0 * Math.PI * i / nFft;
            _cosTable[i] = (float)Math.Cos(angle);
            _sinTable[i] = (float)Math.Sin(angle);
        }

        _melBands = BuildHtkMelFilterbank(sampleRate, nFft, nMels);
    }

    private static MelBand[] BuildHtkMelFilterbank(int sampleRate, int nFft, int nMels)
    {
        int nFreqs = nFft / 2 + 1;
        double fMin = 0.0;
        double fMax = sampleRate / 2.0;

        double melMin = HzToMelHtk(fMin);
        double melMax = HzToMelHtk(fMax);

        var fPts = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            double mel = melMin + (melMax - melMin) * i / (nMels + 1);
            fPts[i] = MelToHzHtk(mel);
        }

        var allFreqs = new double[nFreqs];
        for (int b = 0; b < nFreqs; b++)
        {
            allFreqs[b] = fMax * b / (nFreqs - 1);
        }

        var bands = new MelBand[nMels];
        for (int m = 0; m < nMels; m++)
        {
            double fLeft = fPts[m];
            double fCenter = fPts[m + 1];
            double fRight = fPts[m + 2];
            double leftSpan = fCenter - fLeft;
            double rightSpan = fRight - fCenter;

            int first = -1;
            int last = -1;
            var dense = new float[nFreqs];

            for (int b = 0; b < nFreqs; b++)
            {
                double f = allFreqs[b];
                double down = leftSpan > 0 ? (f - fLeft) / leftSpan : 0.0;
                double up = rightSpan > 0 ? (fRight - f) / rightSpan : 0.0;
                double w = Math.Max(0.0, Math.Min(down, up));
                if (w > 0.0)
                {
                    if (first == -1) first = b;
                    last = b;
                    dense[b] = (float)w;
                }
            }

            if (first == -1)
            {
                bands[m] = new MelBand(0, Array.Empty<float>());
            }
            else
            {
                int len = last - first + 1;
                var weights = new float[len];
                Array.Copy(dense, first, weights, 0, len);
                bands[m] = new MelBand(first, weights);
            }
        }

        return bands;
    }

    private static double HzToMelHtk(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private static double MelToHzHtk(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    /// <summary>
    /// Computes normalized Log-Mel spectrogram flattened row-major as [NMels, TargetFrames].
    /// </summary>
    public float[] Compute(ReadOnlySpan<float> pcmMono)
    {
        int pad = NFft / 2;
        int len = pcmMono.Length;
        int paddedLen = len + 2 * pad;
        var padded = new float[paddedLen];

        // Copy center signal
        pcmMono.CopyTo(padded.AsSpan(pad, len));

        // Reflect padding matching torch.nn.functional.pad(..., mode="reflect")
        if (len > 1)
        {
            for (int i = 0; i < pad; i++)
            {
                int leftSrc = Math.Min(pad - i, len - 1);
                padded[i] = pcmMono[leftSrc];

                int rightSrc = Math.Max(len - 2 - i, 0);
                padded[pad + len + i] = pcmMono[rightSrc];
            }
        }

        int numFrames = Math.Min(TargetFrames, (paddedLen - NFft) / HopLength + 1);
        int nBins = NFft / 2 + 1;

        var real = new float[NFft];
        var imag = new float[NFft];
        var power = new float[nBins];
        var output = new float[NMels * TargetFrames];

        const double amin = 1e-10;
        const float minDb = -100.0f; // 10 * log10(1e-10)

        // Fill default floor in case input is shorter than expected
        Array.Fill(output, minDb);

        for (int t = 0; t < numFrames; t++)
        {
            int offset = t * HopLength;

            // Windowed frame loaded in bit-reversed order for in-place Radix-2 FFT
            for (int i = 0; i < NFft; i++)
            {
                int rev = _bitReverse[i];
                real[rev] = padded[offset + i] * _window[i];
                imag[rev] = 0f;
            }

            RunFftInPlace(real, imag);

            // Power spectrum (|X(k)|^2) for bins 0..nFft/2
            for (int b = 0; b < nBins; b++)
            {
                float re = real[b];
                float im = imag[b];
                power[b] = re * re + im * im;
            }

            // Apply Mel filterbank and convert power to dB
            for (int m = 0; m < NMels; m++)
            {
                var band = _melBands[m];
                double energy = 0.0;
                int start = band.StartBin;
                var weights = band.Weights;

                for (int k = 0; k < weights.Length; k++)
                {
                    energy += power[start + k] * weights[k];
                }

                double db = 10.0 * Math.Log10(Math.Max(energy, amin));
                output[m * TargetFrames + t] = (float)db;
            }
        }

        // Instance normalization: (mel - mean) / (std + 1e-6), using N-1 to match PyTorch tensor.std()
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i];
        double mean = sum / output.Length;

        double sqSum = 0.0;
        for (int i = 0; i < output.Length; i++)
        {
            double d = output[i] - mean;
            sqSum += d * d;
        }
        double std = Math.Sqrt(sqSum / Math.Max(1, output.Length - 1));
        double denom = std + 1e-6;

        for (int i = 0; i < output.Length; i++)
        {
            output[i] = (float)((output[i] - mean) / denom);
        }

        return output;
    }

    private void RunFftInPlace(float[] real, float[] imag)
    {
        int n = NFft;
        for (int size = 2; size <= n; size <<= 1)
        {
            int halfSize = size >> 1;
            int tableStep = n / size;

            for (int i = 0; i < n; i += size)
            {
                int k = 0;
                for (int j = i; j < i + halfSize; j++)
                {
                    int partner = j + halfSize;
                    float cos = _cosTable[k];
                    float sin = _sinTable[k];

                    float tReal = real[partner] * cos - imag[partner] * sin;
                    float tImag = real[partner] * sin + imag[partner] * cos;

                    real[partner] = real[j] - tReal;
                    imag[partner] = imag[j] - tImag;
                    real[j] += tReal;
                    imag[j] += tImag;

                    k += tableStep;
                }
            }
        }
    }
}
