# Tutorial

bar_chart_race offers a wide range of inputs to customize the animation. On this page, we'll cover many of the available options.

## Data format

The data you choose to animate as a bar chart race must be provided in a specific format. The data must be within a pandas DataFrame containing 'wide' data where:

* Each row represents a single period of time
* Each column holds the value for a particular category
* The index contains the time component (optional)

### Example data

Below, we have an example of properly formatted data that shows total deaths from COVID-19 for several countries by date. Each row represents a single day's worth of data. Each column represents a single country's deaths. The index contains the date. Any pandas DataFrame that conforms to this structure may be used to create a bar chart race.

{% include 'html/tutorial_1.html' %}

## Basic bar chart races

A single main function, `bar_chart_race`, exists to create the animations. Calling it with the defaults returns the animation as an HTML string. The `load_dataset` function is available to load sample DataFrames. If you are working within a Jupyter Notebook, it will automatically be embedded in the output as a video.

```python
import bar_chart_race
df = bcr.load_dataset('covid19_tutorial')
df.bcr.bar_chart_race()
```

{% macro video(name) %}
    <div class="vid">
        <video controls ><source src="../videos/{{ name }}.mp4" type="video/mp4"></video>
    </div>
{% endmacro %}

<div>{{ video('basic_default') }}</div>

### Vertical bars

By default, bars are horizontal. Use the `orientation` parameter to switch to vertical.

```python
df.bcr.bar_chart_race(orientation='v')
```

<div>{{ video('basic_vert') }}</div>

### Ascending bars

By default, the bars are plotted in descending order. Change the order by setting `sort` to `'asc'`.

```python
df.bcr.bar_chart_race(sort='asc')
```

<div>{{ video('basic_asc') }}</div>

### Limit the number of bars

By default, all columns will be plotted. Use `n_bars` to limit the number. When limiting bars, the smallest bar can drop off the plot.

```python
df.bcr.bar_chart_race(n_bars=6)
```

<div>{{ video('basic_n_bars') }}</div>

### Fix the order of the bars

By default, the bars will be ordered. Set `fixed_order` to `True` or to a specific list of column names to keep the order the same throughout.

```python
df.bcr.bar_chart_race(fixed_order=['Iran', 'USA', 'Italy', 'Spain', 'Belgium'])
```

<div>{{ video('basic_fixed_order') }}</div>

### Fix the maximum value

By default, the maximum value of the axis moves with the largest bar. Set `fixed_max` to `True` to keep the maximum value equal to the largest overall value for all frames in the animation.

```python
df.bcr.bar_chart_race(fixed_max=True)
```

<div>{{ video('basic_fixed_max') }}</div>

### Change animation smoothness

By default, 10 frames are used to step from one period to the next. Increase/decrease the smoothness of the animation with `steps_per_period`.

```python
df.bcr.bar_chart_race(steps_per_period=3)
```

<div>{{ video('basic_steps') }}</div>

You may also change the amount of time per period with `period_length`, which is set to 500 milliseconds (half of a second) by default.

```python
df.bcr.bar_chart_race(steps_per_period=20, period_length=200)
```

<div>{{ video('basic_period_length') }}</div>

### Interpolate the period

By default, the label for each frame changes after the entire period has been plotted. Linearly interpolate the value for the period with `interpolate_period`. Below, every frame increases by 1 / 10 of a day (2 hours and 24 minutes).

```python
df.bcr.bar_chart_race(interpolate_period=True)
```

<div>{{ video('basic_interpolate') }}</div>

## Plot properties

Many properties of the plot can be set.  

* `figsize` - sets the figure size using matplotlib figure inches (default: `(6, 3.5)`)
* `dpi` - controls the dots per square inch (default: `144`)
* `label_bars` - whether to label the bar values with text (default: `True`)
* `period_label` - dictionary of matplotlib text properties or boolean (default: `True`)
* `title` - title of plot

```python
df.bcr.bar_chart_race(figsize=(5, 3), dpi=100, label_bars=False,
                   period_label={'x': .99, 'y': .1, 'ha': 'right', 'color': 'red'},
                   title='COVID-19 Deaths by Country')
```

<div>{{ video('basic_props') }}</div>

### Label sizes

Control the size of labels with `bar_label_size`, `tick_label_size`, and `title_size`.

```python
df.bcr.bar_chart_race(bar_label_size=4, tick_label_size=5,
                   title='COVID-19 Deaths by Country', title_size='smaller')
```

<div>{{ video('basic_label_size') }}</div>

### Setting font properties

Set font properties for all text objects with `shared_fontdict`.

```python
df.bcr.bar_chart_race(title='COVID-19 Deaths by Country',
                   shared_fontdict={'family': 'Helvetica', 'weight': 'bold',
                                    'color': 'rebeccapurple'})
```

<div>{{ video('basic_shared_font') }}</div>
### Customize bar properties

Set `bar_kwargs` to a dictionary of keyword arguments forwarded to the matploblib `bar` function to control bar properties.

```python
df.bcr.bar_chart_race(bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})
```

<div>{{ video('basic_bar_kwargs') }}</div>

## Additional features

There are several more additional features to customize the animation.

### Formatting the period

Format the label of the period by setting `period_fmt` to a string with either a date directive or a new-style formatted string.

```python
df.bcr.bar_chart_race(period_fmt='%b %-d, %Y')
```

<div>{{ video('other_date_directive') }}</div>

### Use numbers for the index instead of dates

It's not necessary to have dates or times in the index of the DataFrame. Below, the index is dropped, which replaces it with integers beginning at 0. These are then interpolated and formatted.

```python
bcr.bar_chart_race(df.reset_index(drop=True), interpolate_period=True,
                   period_fmt='Index value - {x:.2f}')
```

<div>{{ video('other_string_fmt') }}</div>

### Add text summarizing the entire period

Define a function that accepts two arguments, the values and ranks of the current period of data, and returns
a dictionary that will be passed to the matplotlib `text` function.

```python
def summary(values, ranks):
    total_deaths = int(round(values.sum(), -2))
    s = f'Total Deaths - {total_deaths:,.0f}'
    return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}

df.bcr.bar_chart_race(period_summary_func=summary)
```

<div>{{ video('other_summary') }}</div>

## Add a perpendicular bar

Add a single bar perpendicular to the main bars by defining a function that accepts two arguments, the values and ranks of the current period of data, and returns a single number, the position of the bar. You can use string names of aggregation functions that pandas understands.

```python
df.bcr.bar_chart_race(perpendicular_bar_func='mean')
```

<div>{{ video('other_perpendicular') }}</div>

An example with a user-defined function:

```python
def func(values, ranks):
    return values.quantile(.9)
df.bcr.bar_chart_race(perpendicular_bar_func=func)
```

<div>{{ video('other_perpendicular_func') }}</div>

## Bar colors

By default, the `'dark12'` colormap is used, with 12 unique colors. This is a qualitative color map containing every other color from the 'dark24' colormap originally found from the [plotly express documentation](https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express). All [matplotlib](https://matplotlib.org/tutorials/colors/colormaps.html) and [plotly](https://plotly.com/python/builtin-colorscales/) colormaps are available by name. The entire `'dark24'` colormap will be used by default when your DataFrame contains more than 12 columns.

```python
df.bcr.bar_chart_race(cmap='antique')
```

<div>{{ video('color_map') }}</div>

### Reduce color repetition

It is possible that some colors repeat in your animation, even if there are more colors in the colormap than bars in the animation. This will only happen if you set the `n_bars` parameter, as colors are assigned to each column upon. You'll get a warning advising you to set 
`filter_column_colors` to `True`, which will only assign colors to those bars appearing in the animation.

The following example uses the Accent colormap which has 8 unique colors. The animation is set to have a maximum of 7 bars, but there are still repeating colors.

```python
df.bcr.bar_chart_race(cmap='accent', n_bars=7)
```

!!! warning "`UserWarning`"
    Some of your columns never make an appearance in the animation. To reduce color repetition, set `filter_column_colors` to `True`

<div>{{ video('color_warning') }}</div>

Setting `filter_column_colors` to `True` will reduce the likelihood of repeating colors, but will still happen if the total number of unique bars is more than the number of colors in the colormap.

```python
df.bcr.bar_chart_race(cmap='accent', n_bars=7, filter_column_colors=True)
```

<div>{{ video('color_warning_fixed') }}</div>

## Using your own figure

If you want to highly customize the animation, set the `fig` parameter to a previously created figure. This figure must have at aleast one matplotlib axes created within it.

```python
fig, ax = plt.subplots(figsize=(5, 2), dpi=120)
ax.set_facecolor((0, 0, 1, .3))
df.bcr.bar_chart_race(n_bars=3, fig=fig)
```

<div>{{ video('other_figure') }}</div>

### With subplots

It's possible to add an animation to a matplotlib figure containing multiple subplots. The first subplot will be used for the animation.

```python
from matplotlib import dates
fig, ax_array = plt.subplots(2, 2, figsize=(8, 4), dpi=120, tight_layout=True)
ax1, ax2, ax3, ax4 = ax_array.flatten()
fig.suptitle('Animation in First Axes', y=1)

ax2.plot(df)
ax2.xaxis.set_major_locator(dates.DayLocator([3, 7, 12]))
ax3.bar(df.index, df.median(axis=1))
ax3.xaxis.set_major_locator(dates.DayLocator([3, 7, 12]))
ax4.pie(df.iloc[-1], radius=1.5, labels=df.columns)

df.bcr.bar_chart_race(n_bars=3, fig=fig)
```

<div>{{ video('other_subplots') }}</div>

## Saving the animation

### Default returned values

By default, the video will be embedded into your Jupyter Notebook. If you are not in a Jupyter Notebook, but have IPython installed, an `HTML` object will be returned. Retrieve the underlying HTML with the `data` attribute.

```python
html = bcr.bar_chart_race(df)
html.data # very long string of HTML
```

If you do not have IPython installed, then a string of HTML will be returned directly.

### Saving to disk

In order to save the animation to disk, use a string of the file name of where you'd like to save as the second argument. You'll need to [install ffmpeg](../installation#installing-ffmpeg) first in order to save the animation. Once installed, you'll be able to save the animation as a wide variety of formats (mp4, m4v, mov, etc...). To save the animation as a gif, install ImageMagick.

```
df.bcr.bar_chart_race('docs/videos/covid19.mp4', figsize=(5, 3))
```

### Matplotlib writer

To customize the animation, set the `writer` parameter to a matplotlib `MovieWriter` object instance.