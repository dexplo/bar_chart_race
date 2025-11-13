# Bar Chart Racer - Visual Improvements

## Overview

This document showcases the visual improvements made to bar_chart_racer to provide a more modern, professional appearance while maintaining full backward compatibility.

## Quick Start Example

```python
import bar_chart_racer as bcr

# Load sample data
df = bcr.load_dataset('covid19')

# Create bar chart race with improved visuals (automatic!)
bcr.bar_chart_race(
    df=df,
    filename='output.mp4',
    title='COVID-19 Deaths by Country'
)
```

That's it! The improvements are applied automatically with better defaults.

## What Changed?

### Matplotlib Bar Charts

#### Typography Improvements
| Element | Old Default | New Default | Change |
|---------|-------------|-------------|--------|
| Bar labels | 7pt | **9pt** | +29% larger |
| Tick labels | 7pt | **9pt** | +29% larger |
| Period labels | 12pt | **16pt bold** | +33% larger, bold |
| Period alpha | 1.0 | **0.7** | Subtle transparency |

#### Visual Styling
| Element | Old Default | New Default | Improvement |
|---------|-------------|-------------|-------------|
| Background | `#e5e5e5` | **`#f5f5f5`** | Lighter, cleaner |
| Bar opacity | 0.8 | **0.9** | More solid |
| Bar edge width | 1.0px | **1.5px** | Defined edges |
| Grid lines | Thin | **1.5px + 80% alpha** | Thicker, visible |
| Tick padding | 2px | **4px** | More spacious |
| Tick color | Default | **`#555555`** | Better contrast |

### Plotly Bar Charts

| Element | Old Default | New Default | Change |
|---------|-------------|-------------|--------|
| Font size | 12pt | **13pt** | +8% larger |
| Bar opacity | 0.8 | **0.9** | More solid |

## Visual Comparison

### Before (Old Defaults)
```
├── Small fonts (7pt) - Hard to read
├── Light bars (80% opacity) - Washed out
├── Thin grid lines - Barely visible
├── Small period labels (12pt) - Not prominent
└── Darker background (#e5e5e5) - Heavy feel
```

### After (New Defaults)
```
├── Larger fonts (9pt) - Easy to read ✓
├── Solid bars (90% opacity) - Vibrant ✓
├── Thick grid lines (1.5px + alpha) - Clear guides ✓
├── Bold period labels (16pt) - Eye-catching ✓
└── Light background (#f5f5f5) - Modern, clean ✓
```

## Code Examples

### Example 1: Basic Usage (Improvements Applied Automatically)

```python
import bar_chart_racer as bcr

df = bcr.load_dataset('covid19')

# All improvements are applied by default!
bcr.bar_chart_race(
    df=df,
    filename='covid19.mp4',
    orientation='h',
    sort='desc',
    n_bars=10,
    title='COVID-19 Deaths'
)
```

### Example 2: Custom Styling (Override Defaults)

```python
# You can still customize everything!
bcr.bar_chart_race(
    df=df,
    filename='custom.mp4',

    # Override font sizes if desired
    bar_label_font=12,  # Override new default of 9
    tick_label_font={'size': 11, 'color': 'blue'},

    # Period label customization
    period_label={
        'size': 20,  # Even larger!
        'weight': 'normal',  # Not bold
        'alpha': 1.0  # Fully opaque
    },

    # Bar styling
    bar_kwargs={
        'alpha': 0.95,  # Override new default of 0.9
        'ec': 'black',  # Black edges
        'lw': 2.0  # Thicker edges
    }
)
```

### Example 3: Using Plotly (Interactive)

```python
from bar_chart_racer import bar_chart_race_plotly

# Plotly version with improved defaults
bar_chart_race_plotly(
    df=df,
    filename='interactive.html',
    orientation='h',
    sort='desc',
    n_bars=10,
    title='Interactive COVID-19 Deaths'
)
# Larger fonts and better opacity applied automatically!
```

## Benefits

1. **Better Readability** 📖
   - Larger text makes charts easier to read from a distance
   - Better contrast for improved visibility

2. **Modern Aesthetic** 🎨
   - Lighter background creates a contemporary feel
   - Refined styling matches current design trends

3. **Professional Appearance** 💼
   - Bold, prominent period labels
   - Clean, well-defined bars with proper edges
   - Balanced spacing and padding

4. **Backward Compatible** ✅
   - All existing code continues to work
   - Users can override any default
   - No breaking changes

5. **Zero Effort** 🚀
   - Improvements applied automatically
   - No code changes required
   - Instant visual upgrade

## Technical Details

### Files Modified
- `bar_chart_racer/_bar_chart_race.py` - Matplotlib implementation
- `bar_chart_racer/_bar_chart_race_plotly.py` - Plotly implementation

### Changed Functions
- `get_bar_kwargs()` - Enhanced bar appearance
- `get_period_label()` - Larger, bolder period labels
- `get_font()` - Increased default font sizes
- `prepare_axes()` - Modernized background and grid styling

## Demo Files

Run these demos to see the improvements:

```bash
# Interactive Plotly demo (works without ffmpeg)
python demo_plotly_visuals.py

# Open the generated HTML file in your browser
# Location: demo_improved_visuals.html
```

## Feedback

The visual improvements are designed to provide a better out-of-box experience while maintaining complete flexibility for customization. All previous parameters continue to work exactly as before.
