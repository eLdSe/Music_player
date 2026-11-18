using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SoundPulse.Audio;
using SoundPulse.Domain;

namespace SoundPulse.AI;

public sealed class MusicClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly MelSpectrogram _melProcessor;
    private readonly object _inferenceLock = new();
    private bool _disposed;

    public ModelLabels Labels { get; }

    public MusicClassifier()
    {
        var assembly = Assembly.GetExecutingAssembly();

        Labels = LoadEmbeddedLabels(assembly);
        byte[] modelBytes = LoadEmbeddedResourceBytes(assembly, "music_classifier.onnx");

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Clamp(Environment.ProcessorCount / 2, 1, 4),
            InterOpNumThreads = 1
        };

        _session = new InferenceSession(modelBytes, options);
        _melProcessor = new MelSpectrogram(
            sampleRate: Labels.SampleRate,
            nFft: Labels.NFft,
            hopLength: Labels.HopLength,
            nMels: Labels.NMels,
            chunkSamples: Labels.ChunkSamples);
    }

    private static ModelLabels LoadEmbeddedLabels(Assembly assembly)
    {
        using var stream = assembly.GetManifestResourceStream("labels.json")
            ?? throw new InvalidOperationException("Embedded resource 'labels.json' not found.");
        return JsonSerializer.Deserialize<ModelLabels>(stream)
            ?? throw new InvalidOperationException("Failed to deserialize embedded 'labels.json'.");
    }

    private static byte[] LoadEmbeddedResourceBytes(Assembly assembly, string name)
    {
        using var stream = assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"Embedded resource '{name}' not found.");
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return ms.ToArray();
    }

    public AnalysisResult AnalyzeFile(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var (chunks, durationSec) = AudioDecoder.DecodeRepresentativeChunks16kMono(
            filePath,
            Labels.SampleRate,
            Labels.ChunkDurationSec);

        return ClassifyMultiCrop(chunks, durationSec);
    }

    public AnalysisResult ClassifyMultiCrop(IReadOnlyList<float[]> chunks, double durationSec)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (chunks.Count == 0)
            throw new InvalidOperationException("No audio samples decoded.");

        var sw = Stopwatch.StartNew();

        double[] sumGenreLogits = new double[Labels.Genres.Count];
        double[] sumVibeLogits = new double[Labels.Vibes.Count];

        foreach (var pcm in chunks)
        {
            float[] melFlat = _melProcessor.Compute(pcm);
            var inputTensor = new DenseTensor<float>(
                melFlat,
                [1, 1, Labels.NMels, _melProcessor.TargetFrames]);

            var inputs = new List<NamedOnnxValue>(1)
            {
                NamedOnnxValue.CreateFromTensor("mel_spectrogram", inputTensor)
            };

            float[] gLogits;
            float[] vLogits;

            lock (_inferenceLock)
            {
                using var results = _session.Run(inputs);
                var list = results.ToList();
                gLogits = list[0].AsEnumerable<float>().ToArray();
                vLogits = list[1].AsEnumerable<float>().ToArray();
            }

            for (int i = 0; i < Math.Min(sumGenreLogits.Length, gLogits.Length); i++)
                sumGenreLogits[i] += gLogits[i];
            for (int i = 0; i < Math.Min(sumVibeLogits.Length, vLogits.Length); i++)
                sumVibeLogits[i] += vLogits[i];
        }

        sw.Stop();

        float invN = 1f / chunks.Count;
        float[] avgGenreLogits = new float[sumGenreLogits.Length];
        float[] avgVibeLogits = new float[sumVibeLogits.Length];
        for (int i = 0; i < avgGenreLogits.Length; i++) avgGenreLogits[i] = (float)(sumGenreLogits[i] * invN);
        for (int i = 0; i < avgVibeLogits.Length; i++) avgVibeLogits[i] = (float)(sumVibeLogits[i] * invN);

        float[] genreProbs = Softmax(avgGenreLogits);
        float[] onnxVibeProbs = Softmax(avgVibeLogits);

        // Hybrid DSP feature extraction (BPM, Energy, Percussiveness, Brightness)
        var dsp = AudioEnergyAnalyzer.Analyze(chunks);
        // For v1 (3-class GTZAN model), blend DSP energy to prevent ballad->drive bias.
        // For v2+ (5-class CLAP-distilled mood model), trust the model's dedicated mood head directly!
        float[] vibeProbs = (Labels.Version >= 2 || Labels.Vibes.Count != 3)
            ? onnxVibeProbs
            : FuseVibeWithDsp(onnxVibeProbs, genreProbs, dsp);

        var genreScores = new List<ScoreEntry>(Labels.Genres.Count);
        for (int i = 0; i < Math.Min(Labels.Genres.Count, genreProbs.Length); i++)
        {
            string key = Labels.Genres[i];
            genreScores.Add(new ScoreEntry(
                key,
                CategoryPalette.FormatGenreName(key),
                genreProbs[i],
                CategoryPalette.GetGenreColor(key)));
        }
        genreScores.Sort((a, b) => b.Probability.CompareTo(a.Probability));

        var vibeScores = new List<ScoreEntry>(Labels.Vibes.Count);
        int bestVibeIdx = 0;
        float bestVibeProb = -1f;
        for (int i = 0; i < Math.Min(Labels.Vibes.Count, vibeProbs.Length); i++)
        {
            if (vibeProbs[i] > bestVibeProb)
            {
                bestVibeProb = vibeProbs[i];
                bestVibeIdx = i;
            }
            vibeScores.Add(new ScoreEntry(
                i.ToString(),
                Labels.Vibes[i],
                vibeProbs[i],
                CategoryPalette.GetVibeColor(i, Labels.Vibes[i])));
        }

        var topGenre = genreScores[0];

        return new AnalysisResult
        {
            Genre = topGenre.Key,
            GenreConfidence = topGenre.Probability,
            Vibe = Labels.Vibes[bestVibeIdx],
            VibeIndex = bestVibeIdx,
            VibeConfidence = bestVibeProb,
            TopGenres = genreScores.Take(5).ToList(),
            VibeScores = vibeScores,
            DurationSec = durationSec,
            InferenceMs = sw.Elapsed.TotalMilliseconds,
            Bpm = dsp.Bpm,
            Energy = dsp.Energy
        };
    }

    /// <summary>
    /// Fuses MusicAudioNet's timbre-based vibe probabilities with physical DSP energy/tempo features:
    /// - High energy + fast/punchy beat -> boosts Drive (0)
    /// - Moderate/soft energy or slow lyrical tempo -> boosts Chill (1), prevents slow ballads from being Drive
    /// - Calm acoustic/low-percussive texture + classical/jazz timbre -> boosts Focus (2)
    /// </summary>
    private static float[] FuseVibeWithDsp(float[] onnxVibe, float[] genreProbs, AudioDspFeatures dsp)
    {
        if (onnxVibe.Length < 3) return onnxVibe;

        float onnxDrive = onnxVibe[0];
        float onnxChill = onnxVibe[1];
        float onnxFocus = onnxVibe[2];

        // Classical (idx 1) + Jazz (idx 5) probability from ONNX
        float instrumentalPrior = (genreProbs.Length > 5) ? (genreProbs[1] + genreProbs[5]) : onnxFocus;

        // DSP Drive evidence: strong energy + upbeat tempo + punchy beat
        float tempoDrive = Math.Clamp((dsp.Bpm - 92f) / 38f, 0f, 1f);
        float dspDrive = Math.Clamp(
            dsp.Energy * 0.45f +
            dsp.Percussiveness * 0.30f +
            tempoDrive * 0.25f,
            0.02f, 1f);

        // DSP Focus evidence: low percussiveness, low brightness harshness, calm acoustic texture
        float calmness = Math.Clamp(1f - dsp.Percussiveness * 0.65f - dsp.Energy * 0.35f, 0f, 1f);
        float dspFocus = Math.Clamp(calmness * 0.65f + instrumentalPrior * 0.55f, 0.02f, 1f);

        // DSP Chill evidence: mid/slow tempo or soft/warm dynamics (lyrical ballads, mellow pop, acoustic)
        float slowOrMidTempo = Math.Clamp(1.15f - Math.Abs(dsp.Bpm - 88f) / 48f, 0f, 1f);
        float moderateEnergy = Math.Clamp(1f - Math.Abs(dsp.Energy - 0.46f) / 0.48f, 0f, 1f);
        float softBeat = Math.Clamp(1f - dsp.Percussiveness * 0.7f, 0.1f, 1f);
        float dspChill = Math.Clamp(
            slowOrMidTempo * 0.42f +
            moderateEnergy * 0.33f +
            softBeat * 0.25f,
            0.02f, 1f);

        // Ballad damper: if tempo is clearly slow (< 98 BPM) and percussiveness isn't aggressive,
        // dampen GTZAN's rock->drive bias so emotional ballads land in Chill instead of Drive
        float driveDamp = (dsp.Bpm < 98 && dsp.Percussiveness < 0.62f)
            ? Math.Clamp(0.45f + (dsp.Bpm - 65f) / 65f, 0.40f, 0.85f)
            : 1.0f;

        float wDrive = (onnxDrive * 0.48f + dspDrive * 0.52f) * driveDamp;
        float wChill = onnxChill * 0.45f + dspChill * 0.55f + (1f - driveDamp) * onnxDrive * 0.65f;
        float wFocus = onnxFocus * 0.58f + dspFocus * 0.42f;

        // Sharpen slightly and normalize to valid probability distribution
        float sum = wDrive + wChill + wFocus;
        if (sum <= 1e-6f) return onnxVibe;

        return [wDrive / sum, wChill / sum, wFocus / sum];
    }

    private static float[] Softmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        double sum = 0.0;
        var probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            double e = Math.Exp(logits[i] - max);
            probs[i] = (float)e;
            sum += e;
        }

        if (sum > 0.0)
        {
            for (int i = 0; i < probs.Length; i++)
                probs[i] = (float)(probs[i] / sum);
        }

        return probs;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}
