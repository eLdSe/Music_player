using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using SoundPulse.AI;
using SoundPulse.UI;

namespace SoundPulse;

public partial class App : Application
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AttachConsole(uint dwProcessId);

    private const uint AttachParentProcess = unchecked((uint)-1);

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Optional headless verification mode: SoundPulse.exe --self-test <audioFile>
        if (e.Args.Length >= 2 && e.Args[0].Equals("--self-test", StringComparison.OrdinalIgnoreCase))
        {
            RunSelfTest(e.Args[1]);
            Shutdown(0);
            return;
        }

        DispatcherUnhandledException += (_, args) =>
        {
            MessageBox.Show(
                $"Произошла ошибка:\n{args.Exception.Message}",
                "SoundPulse AI",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            args.Handled = true;
        };

        var mainWindow = new MainWindow(e.Args);
        MainWindow = mainWindow;
        mainWindow.Show();
    }

    private static void RunSelfTest(string audioPath)
    {
        _ = AttachConsole(AttachParentProcess);
        try
        {
            using var classifier = new MusicClassifier();
            var res = classifier.AnalyzeFile(audioPath);
            string summary =
                $"FILE: {Path.GetFileName(audioPath)}\n" +
                $"GENRE: {res.Genre} ({res.GenreConfidence * 100f:0.0}%)\n" +
                $"VIBE: {res.Vibe} ({res.VibeConfidence * 100f:0.0}%)\n" +
                $"DSP: {res.Bpm} BPM, Energy={res.Energy * 100f:0.0}%\n" +
                $"INFERENCE_MS: {res.InferenceMs:0.0}\n" +
                $"TOP3: " + string.Join(", ", res.TopGenres.Take(3).Select(g => $"{g.DisplayName}={g.PercentText}"));

            Console.WriteLine(summary);
            string logPath = Path.Combine(Path.GetTempPath(), "soundpulse_selftest.txt");
            File.WriteAllText(logPath, summary);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"SELF-TEST ERROR: {ex}");
            string logPath = Path.Combine(Path.GetTempPath(), "soundpulse_selftest.txt");
            File.WriteAllText(logPath, $"ERROR: {ex}");
            Environment.ExitCode = 1;
        }
    }
}
