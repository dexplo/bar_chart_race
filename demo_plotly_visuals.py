#!/usr/bin/env python3
"""
Demo script showcasing the improved Plotly bar chart visuals
"""

import bar_chart_racer as bcr
import pandas as pd

print("=" * 70)
print("BAR CHART RACER - VISUAL IMPROVEMENTS DEMO (Plotly)")
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
print("GENERATING PLOTLY BAR CHART RACE WITH IMPROVED VISUALS")
print("=" * 70)

print("\n✨ Visual Improvements Applied:")
print("  • Larger fonts (13pt default)")
print("  • Smoother bars with 90% opacity (was 80%)")
print("  • Better readability and contrast")
print("  • Interactive Plotly features")

# Create the animation using Plotly
print("\nCreating Plotly animation...")

from bar_chart_racer import bar_chart_race_plotly

output_file = '/home/user/bar_chart_racer/demo_improved_visuals.html'

bar_chart_race_plotly(
    df=df,
    filename=output_file,
    orientation='h',
    sort='desc',
    n_bars=10,
    steps_per_period=5,
    period_length=300,
    title='COVID-19 Deaths - Visual Improvements Demo',
    bar_size=.95,
)

print(f"\n✅ Animation created successfully!")
print(f"\n💾 Saved to: {output_file}")

print("\n" + "=" * 70)
print("COMPARING OLD VS NEW DEFAULTS")
print("=" * 70)

print("\n📊 Plotly Visual Improvements:")
print("   Old → New")
print("   Font size:     12pt → 13pt  (+8% larger)")
print("   Bar opacity:   0.8  → 0.9   (more solid)")

print("\n📊 Matplotlib Visual Improvements:")
print("   Old → New")
print("   Bar labels:     7pt  → 9pt  (+29% larger)")
print("   Tick labels:    7pt  → 9pt  (+29% larger)")
print("   Period labels: 12pt  → 16pt (+33% larger, now bold)")
print("   Background:    #e5e5e5 → #f5f5f5 (lighter)")
print("   Bar opacity:   0.8     → 0.9 (more solid)")
print("   Bar edge:      1.0px   → 1.5px (defined)")
print("   Grid lines:    thin    → 1.5px + transparency")

print("\n" + "=" * 70)
print("✨ The improvements provide a more modern, professional look")
print("   while maintaining full backward compatibility!")
print("=" * 70)

print("\n📖 To view the animation:")
print(f"   Open {output_file} in your web browser")
