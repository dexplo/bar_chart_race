#!/usr/bin/env python3
"""
Demo script showcasing the improved bar chart visuals
"""

import bar_chart_racer as bcr
import pandas as pd

print("=" * 70)
print("BAR CHART RACER - VISUAL IMPROVEMENTS DEMO")
print("=" * 70)

# Load sample data
df = bcr.load_dataset('covid19')
df = df.iloc[-10:]  # Last 10 time periods for a quick demo

print(f"\nLoaded COVID-19 dataset: {df.shape[0]} time periods, {df.shape[1]} countries")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Show a preview of the data
print("\nData preview:")
print(df.tail(3))

print("\n" + "=" * 70)
print("GENERATING BAR CHART RACE WITH IMPROVED VISUALS")
print("=" * 70)

print("\n✨ Visual Improvements Applied:")
print("  • Larger fonts (9pt for labels, 16pt for period labels)")
print("  • Bold period labels with subtle transparency")
print("  • Modern light background (#f5f5f5)")
print("  • Enhanced grid with thicker white lines")
print("  • Smoother bars with 90% opacity")
print("  • Better contrast and spacing")

# Create the animation (returns HTML since no filename provided)
print("\nCreating animation...")

html = bcr.bar_chart_race(
    df=df,
    filename=None,  # Returns HTML
    orientation='h',
    sort='desc',
    n_bars=10,
    steps_per_period=5,
    period_length=300,
    interpolate_period=False,
    period_label=True,
    period_template='%B %d, %Y',
    title='COVID-19 Deaths - Visual Improvements Demo',
    bar_size=.95,
    bar_textposition='outside',
    scale='linear'
)

print("\n✅ Animation created successfully!")
print(f"\nHTML output size: {len(html):,} characters")

# Save to file to view in browser
output_file = '/home/user/bar_chart_racer/demo_improved_visuals.html'
with open(output_file, 'w') as f:
    f.write(html)

print(f"\n💾 Saved to: {output_file}")
print("\nTo view the animation, open the HTML file in a web browser:")
print(f"   {output_file}")

print("\n" + "=" * 70)
print("COMPARING OLD VS NEW DEFAULTS")
print("=" * 70)

print("\n📊 Typography:")
print("   Old → New")
print("   Bar labels:     7pt  → 9pt  (+29% larger)")
print("   Tick labels:    7pt  → 9pt  (+29% larger)")
print("   Period labels: 12pt  → 16pt (+33% larger, now bold)")

print("\n🎨 Visual Styling:")
print("   Old → New")
print("   Background:    #e5e5e5 → #f5f5f5 (lighter)")
print("   Bar opacity:   0.8     → 0.9 (more solid)")
print("   Bar edge:      1.0px   → 1.5px (defined)")
print("   Grid lines:    thin    → 1.5px + transparency")
print("   Tick padding:  2px     → 4px (more spacious)")

print("\n" + "=" * 70)
print("✨ The improvements provide a more modern, professional look")
print("   while maintaining full backward compatibility!")
print("=" * 70)
