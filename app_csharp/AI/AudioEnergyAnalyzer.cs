namespace SoundPulse.AI;

public readonly record struct AudioDspFeatures(
    int Bpm,
    float Energy,
    float Percussiveness,
    float Brightness,
    float TempoStrength);

/// <summary>
/// Fast pure-C# DSP analyzer estimating Tempo (BPM), Energy, Percussiveness (onset flux),
/// and Brightness directly from 16 kHz mono PCM chunks (~2-4 ms total).
/// Complements MusicAudioNet so slow lyrical ballads aren't misclassified as high-energy "Drive".
/// </summary>
public static class AudioEnergyAnalyzer
{
    private const int SampleRate = 16000;
    private const int FrameSize = 1024;
    private const int HopSize = 512;

    public static AudioDspFeatures Analyze(IReadOnlyList<float[]> chunks)
    {
        if (chunks.Count == 0)
            return new AudioDspFeatures(100, 0.5f, 0.5f, 0.5f, 0.5f);

        double sumBpm = 0;
        double sumBpmWeight = 0;
        double sumEnergy = 0;
        double sumPerc = 0;
        double sumBright = 0;
        double sumTempoStr = 0;

        foreach (var pcm in chunks)
        {
            var f = AnalyzeSingleChunk(pcm);
            double w = Math.Max(0.15, f.TempoStrength);
            sumBpm += f.Bpm * w;
            sumBpmWeight += w;
            sumEnergy += f.Energy;
            sumPerc += f.Percussiveness;
            sumBright += f.Brightness;
            sumTempoStr += f.TempoStrength;
        }

        int n = chunks.Count;
        int bpm = (int)Math.Round(sumBpm / Math.Max(1e-6, sumBpmWeight));
        return new AudioDspFeatures(
            Bpm: Math.Clamp(bpm, 55, 190),
            Energy: Math.Clamp((float)(sumEnergy / n), 0f, 1f),
            Percussiveness: Math.Clamp((float)(sumPerc / n), 0f, 1f),
            Brightness: Math.Clamp((float)(sumBright / n), 0f, 1f),
            TempoStrength: Math.Clamp((float)(sumTempoStr / n), 0f, 1f));
    }

    private static AudioDspFeatures AnalyzeSingleChunk(float[] pcm)
    {
        int numFrames = Math.Max(1, (pcm.Length - FrameSize) / HopSize + 1);
        var frameRms = new float[numFrames];
        var lowBand = new float[numFrames];
        var highBand = new float[numFrames];
        var onset = new float[numFrames];

        // Simple 1-pole low-pass (~250 Hz) & high-pass (~2000 Hz) filters to separate bass kick vs treble/hi-hats
        // alpha = dt / (rc + dt)
        const float lpAlpha = 0.09f;  // ~250 Hz @ 16 kHz
        const float hpAlpha = 0.56f;  // ~2000 Hz @ 16 kHz

        float lpState = 0f;
        float hpState = 0f;
        float prevSample = 0f;

        double totalSq = 0;
        double totalHighSq = 0;
        long totalZcr = 0;

        for (int t = 0; t < numFrames; t++)
        {
            int start = t * HopSize;
            int end = Math.Min(pcm.Length, start + FrameSize);
            int count = Math.Max(1, end - start);

            double sq = 0;
            double loSq = 0;
            double hiSq = 0;
            int zcr = 0;

            for (int i = start; i < end; i++)
            {
                float x = pcm[i];
                sq += x * x;

                if (i > start && ((x >= 0f) != (pcm[i - 1] >= 0f)))
                    zcr++;

                lpState += lpAlpha * (x - lpState);
                loSq += lpState * lpState;

                hpState = hpAlpha * (hpState + x - prevSample);
                prevSample = x;
                hiSq += hpState * hpState;
            }

            frameRms[t] = MathF.Sqrt((float)(sq / count));
            lowBand[t] = MathF.Sqrt((float)(loSq / count));
            highBand[t] = MathF.Sqrt((float)(hiSq / count));

            totalSq += sq;
            totalHighSq += hiSq;
            totalZcr += zcr;

            if (t > 0)
            {
                // Multi-band half-wave rectified onset flux (captures kick drum + snare + hi-hat attacks)
                float dFull = MathF.Max(0f, frameRms[t] - frameRms[t - 1]);
                float dLow = MathF.Max(0f, lowBand[t] - lowBand[t - 1]);
                float dHigh = MathF.Max(0f, highBand[t] - highBand[t - 1]);
                onset[t] = dFull * 0.45f + dLow * 0.35f + dHigh * 0.20f;
            }
        }

        float overallRms = MathF.Sqrt((float)(totalSq / Math.Max(1, pcm.Length)));
        float highRms = MathF.Sqrt((float)(totalHighSq / Math.Max(1, pcm.Length)));

        // Brightness: ratio of high-frequency energy + ZCR
        float highRatio = highRms / (overallRms + 1e-4f);
        float zcrRate = (float)totalZcr / Math.Max(1, pcm.Length);
        float brightness = Math.Clamp(highRatio * 0.95f + zcrRate * 1.4f, 0f, 1f);

        // Onset sharpness / percussiveness: mean onset relative to signal RMS + peakiness
        double onsetSum = 0;
        double onsetSq = 0;
        for (int t = 1; t < numFrames; t++)
        {
            onsetSum += onset[t];
            onsetSq += onset[t] * onset[t];
        }
        float onsetMean = (float)(onsetSum / Math.Max(1, numFrames - 1));
        float onsetStd = MathF.Sqrt(MathF.Max(0f, (float)(onsetSq / Math.Max(1, numFrames - 1)) - onsetMean * onsetMean));

        // Normalize percussiveness: drums/electronic have strong sharp onset flux (~0.08..0.22 of RMS)
        float relOnset = (onsetMean + onsetStd * 0.8f) / (overallRms + 0.02f);
        float percussiveness = Math.Clamp((relOnset - 0.14f) / 0.34f, 0f, 1f);

        // Tempo (BPM) estimation via autocorrelation of onset envelope
        var (bpm, tempoStrength) = EstimateBpm(onset);

        // Loudness density (RMS in dBFS mapped -32 dB .. -7 dB -> 0..1)
        float rmsDb = 20f * MathF.Log10(MathF.Max(overallRms, 1e-5f));
        float loudnessScore = Math.Clamp((rmsDb + 32f) / 24f, 0f, 1f);

        // Tempo speed factor (70 BPM -> 0.1, 100 BPM -> 0.48, 128+ BPM -> 0.95)
        float tempoFactor = Math.Clamp((bpm - 66f) / 68f, 0f, 1f);

        // Combined perceptual Energy (0..1)
        float energy = Math.Clamp(
            loudnessScore * 0.30f +
            percussiveness * 0.32f +
            tempoFactor * 0.23f +
            brightness * 0.15f,
            0f, 1f);

        return new AudioDspFeatures(bpm, energy, percussiveness, brightness, tempoStrength);
    }

    private static (int Bpm, float Strength) EstimateBpm(float[] onset)
    {
        int n = onset.Length;
        if (n < 40) return (100, 0f);

        // Zero-mean onset signal
        double mean = 0;
        for (int i = 0; i < n; i++) mean += onset[i];
        mean /= n;

        var centered = new float[n];
        double energy0 = 0;
        for (int i = 0; i < n; i++)
        {
            float v = (float)(onset[i] - mean);
            centered[i] = v;
            energy0 += v * v;
        }
        if (energy0 < 1e-8) return (90, 0f);

        // Frame rate = 16000 / 512 = 31.25 fps
        const double fps = (double)SampleRate / HopSize;
        // Lags covering ~60 BPM (lag 31) to ~187 BPM (lag 10)
        const int minLag = 10;
        const int maxLag = 32;

        double bestScore = -1.0;
        int bestLag = 16;
        double bestRawAcf = 0.0;

        for (int lag = minLag; lag <= maxLag && lag < n - 4; lag++)
        {
            double dot = 0;
            for (int i = 0; i < n - lag; i++)
                dot += centered[i] * centered[i + lag];

            double acf = dot / energy0;
            double candBpm = 60.0 * fps / lag;

            // Gentle perceptual prior favoring typical musical tempi (75..140 BPM)
            double logDist = Math.Log2(candBpm / 112.0);
            double prior = Math.Exp(-0.5 * (logDist / 0.65) * (logDist / 0.65));

            double score = acf * prior;
            if (score > bestScore)
            {
                bestScore = score;
                bestLag = lag;
                bestRawAcf = Math.Max(0.0, acf);
            }
        }

        // Parabolic interpolation around bestLag for sub-frame BPM accuracy
        double refinedLag = bestLag;
        if (bestLag > minLag && bestLag < maxLag)
        {
            double rPrev = ComputeAcf(centered, bestLag - 1, energy0);
            double rCurr = ComputeAcf(centered, bestLag, energy0);
            double rNext = ComputeAcf(centered, bestLag + 1, energy0);
            double denom = 2.0 * (rPrev - 2.0 * rCurr + rNext);
            if (Math.Abs(denom) > 1e-6)
            {
                double delta = Math.Clamp((rPrev - rNext) / denom, -0.5, 0.5);
                refinedLag = bestLag + delta;
            }
        }

        int bpm = (int)Math.Round(60.0 * fps / refinedLag);
        float strength = Math.Clamp((float)(bestRawAcf * 2.2), 0f, 1f);
        return (Math.Clamp(bpm, 58, 188), strength);
    }

    private static double ComputeAcf(float[] centered, int lag, double energy0)
    {
        double dot = 0;
        int limit = centered.Length - lag;
        for (int i = 0; i < limit; i++)
            dot += centered[i] * centered[i + lag];
        return dot / energy0;
    }
}
