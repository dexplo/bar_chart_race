# What's New

## Version 0.2

Upcoming release on July xx, 2020

### Major New Features

* Plotly animated bar charts with `bar_chart_race_plotly`
* Line chart races with `line_chart_race`

#### Other enhancements

* Integration directly into pandas DataFrames - `df.bcr.bar_chart_race`
* Bar label position able to be specified ('outside', 'inside', or None) using new parameter `bar_textposition`
* Bar label formatting possible with string or function using new parameter `bar_texttemplate`
* Added `end_period_pause` parameter that creates a pause (in milliseconds) at the end of each period
* Parameter `title`, in addition to a string, can also be a dictionary using `'label'` as the key for the title. Other keys may be used to control text properties
* Removed parameters `figsize` and `dpi` in favor of `fig_kwargs` dictionary capable of taking all matplotlib `Figure` parameters
* Figure background color able to be saved
* Several parameters changed name and order

## Version 0.1

Released June 1, 2020

This is the first major release of bar_chart_race adding many features:

* [Fixed bar position](../tutorial#fix-the-order-of-the-bars)
* [Fixed max value](../tutorial#fix-the-maximum-value)
* [Perpendicular bars](../tutorial#add-a-perpendicular-bar)
* [Interpolation of the period](../tutorial#change-animation-smoothness)
* [Formatting of the period label](../tutorial#formatting-the-period)
* [Period label summary](../tutorial#add-text-summarizing-the-entire-period)
* [Support for plotly colormaps](../tutorial#bar-colors)

## Version 0.0.1

Released April 29, 2020

Genesis of bar_chart_race capable of producing smoothly transitioning bars with matplotlib and pandas.