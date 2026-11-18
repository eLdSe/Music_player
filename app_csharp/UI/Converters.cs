using System.Collections.Concurrent;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace SoundPulse.UI;

/// <summary>
/// Converts a hex color string (e.g. "#F43F5E") into a frozen cached SolidColorBrush.
/// Zero allocations after first lookup.
/// </summary>
public sealed class HexToBrushConverter : IValueConverter
{
    private static readonly ConcurrentDictionary<string, SolidColorBrush> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly SolidColorBrush Fallback = CreateFrozen(Color.FromRgb(129, 140, 248));

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string hex || string.IsNullOrWhiteSpace(hex))
            return Fallback;

        return Cache.GetOrAdd(hex, static h =>
        {
            try
            {
                var color = (Color)ColorConverter.ConvertFromString(h);
                return CreateFrozen(color);
            }
            catch
            {
                return Fallback;
            }
        });
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => Binding.DoNothing;

    private static SolidColorBrush CreateFrozen(Color c)
    {
        var b = new SolidColorBrush(c);
        b.Freeze();
        return b;
    }
}

/// <summary>
/// Converts a hex color string into a low-opacity tinted frozen SolidColorBrush for badges/pills.
/// </summary>
public sealed class HexToSoftBrushConverter : IValueConverter
{
    private static readonly ConcurrentDictionary<string, SolidColorBrush> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly SolidColorBrush Fallback = CreateFrozen(Color.FromArgb(40, 129, 140, 248));

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string hex || string.IsNullOrWhiteSpace(hex))
            return Fallback;

        byte alpha = 38;
        if (parameter is string s && byte.TryParse(s, out byte parsed))
            alpha = parsed;

        string key = $"{hex}:{alpha}";
        return Cache.GetOrAdd(key, _ =>
        {
            try
            {
                var c = (Color)ColorConverter.ConvertFromString(hex);
                return CreateFrozen(Color.FromArgb(alpha, c.R, c.G, c.B));
            }
            catch
            {
                return Fallback;
            }
        });
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => Binding.DoNothing;

    private static SolidColorBrush CreateFrozen(Color c)
    {
        var b = new SolidColorBrush(c);
        b.Freeze();
        return b;
    }
}

public sealed class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool flag = value is bool b && b;
        if (Invert) flag = !flag;
        return flag ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => Binding.DoNothing;
}

public sealed class IntToVisibilityConverter : IValueConverter
{
    public bool ShowWhenZero { get; set; }

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        int n = value is int i ? i : 0;
        bool visible = ShowWhenZero ? n == 0 : n > 0;
        return visible ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => Binding.DoNothing;
}
