# Tutorial

bar_chart_race offers a wide range of inputs to customize the animation. On this page, we'll cover many of the available options.

## Data format

The data you choose to animate as a bar chart race must be provided in a specific format. The data must be within a pandas DataFrame containing 'wide' data where:

* Each row represents a single period of time
* Each column holds the value for a particular category
* The index contains the time component (optional)

### Example data

The data below is an example of properly formatted data. It shows total deaths from COVID-19 for several countries by date. Each row represents a single day's worth of data. Each column represents a single country's deaths. The index contains the date. Any pandas DataFrame that conforms to this structure may be used to create a bar chart race.

![](images/wide_data.png)

## Basic bar chart races

A single main function, `bar_chart_race`, exists to create the animations. Calling it with the defaults returns the animation as an HTML string. The `load_dataset` function is available to load sample DataFrames. If you are working within a Jupyter Notebook, you can import the `HTML` function from the `IPython.display` module to embed the video directly into the notebook.

```python
import bar_chart_race as bcr
from IPython.display import HTML
df = bcr.load_dataset('covid19_tutorial')
html_string = bcr.bar_chart_race(df)
HTML(html_string)
```

<style>
    .vid {
        display: flex;
        justify-content: center;
    }
    video {
        width: 85%;
    }
</style>

<div class="vid">
    <video controls ><source src="../videos/basic_default.mp4" type="video/mp4"></video>
</div>

### Vertical bars

By default, bars are horizontal. Use the `orientation` parameter to switch it to vertical.

```python
bcr.bar_chart_race(df, orientation='v')
```

<div class="vid">
    <video controls ><source src="../videos/basic_vert.mp4" type="video/mp4"></video>
</div>

### Ascending bars

By default, the bars are plotted in descending order. Change the order by setting `sort` to `'asc'`.

```python
bcr.bar_chart_race(df, sort='asc')
```

<div class="vid">
    <video controls ><source src="../videos/basic_asc.mp4" type="video/mp4"></video>
</div>

### Limit the number of bars

By default, all columns will be plotted. Use `n_bars` to limit the number. When limiting bars, the smallest bar can drop off the plot.

```python
bcr.bar_chart_race(df, n_bars=6)
```

<div class="vid">
    <video controls ><source src="../videos/basic_n_bars.mp4" type="video/mp4"></video>
</div>

### Fix the order of the bars

By default, the bars will be ordered. Set `fixed_order` to `True` or to a specific list of column names to keep the order the same throughout.

```python
bcr.bar_chart_race(df, fixed_order=['Iran', 'USA', 'Italy', 'Spain', 'Belgium'])
```

<div class="vid">
    <video controls ><source src="../videos/basic_fixed_order.mp4" type="video/mp4"></video>
</div>

### Fix the maximum value

By default, the maximum value of the axis moves with the largest bar. Set `fixed_max` to `True` to keep the maximum value equal to the largest overall value for all frames in the animation.

```python
bcr.bar_chart_race(df, fixed_max=True)
```

<div class="vid">
    <video controls ><source src="../videos/basic_fixed_max.mp4" type="video/mp4"></video>
</div>

### Change animation smoothness

By default, 10 frames are used to step from one period to the next. Increase/decrease the smoothness of the animation with `steps_per_period`.

```python
bcr.bar_chart_race(df, steps_per_period=3)
```

<div class="vid">
    <video controls ><source src="../videos/basic_steps.mp4" type="video/mp4"></video>
</div>

You may also change the amount of time per period with `period_length`, which is set to 500 milliseconds (half of a second) by default.

```python
bcr.bar_chart_race(df, steps_per_period=20, period_length=200)
```

<div class="vid">
    <video controls ><source src="../videos/basic_period_length.mp4" type="video/mp4"></video>
</div>


### Interpolate the period

By default, the label for each frame changes after the entire period has been plotted. Linearly interpolate the value for the period with `interpolate_period`. Below, every frame increase by 1 / 10 of a day (2 hours and 24 minutes).

```python
bcr.bar_chart_race(df, interpolate_period=True)
```

<div class="vid">
    <video controls ><source src="../videos/basic_interpolate.mp4" type="video/mp4"></video>
</div>

## Plot properties

Many properties of the plot can be set.  

* `figsize` - sets the figure size using matplotlib figure inches (default: `(6, 3.5)`)
* `dpi` - controls the dots per square inch (default: `144`)
* `label_bars` - whether to label the bar values with text (default: `True`)
* `period_label` - dictionary of matplotlib text properties or boolean (default: `True`)
* `title` - title of plot

```python
bcr.bar_chart_race(df, figsize=(5, 3), dpi=100, label_bars=False, 
                   period_label={'x': .99, 'y': .1, 'ha': 'right', 'color': 'red'},
                   title='COVID-19 Deaths by Country')
```

<div class="vid">
    <video controls ><source src="../videos/basic_props.mp4" type="video/mp4"></video>
</div>

### Label sizes

Control the size of labels with `bar_label_size`, `tick_label_size`, and `title_size`.

```python
bcr.bar_chart_race(df, bar_label_size=4, tick_label_size=5, 
                   title='COVID-19 Deaths by Country', title_size='smaller')
```

<div class="vid">
    <video controls ><source src="../videos/basic_label_size.mp4" type="video/mp4"></video>
</div>

### Setting font properties

Set font properties for all text objects with `shared_fontdict`.

```python
bcr.bar_chart_race(df, title='COVID-19 Deaths by Country',
                   shared_fontdict={'family': 'Helvetica', 'weight': 'bold',
                                    'color': 'rebeccapurple'})
```

<div class="vid">
    <video controls ><source src="../videos/basic_shared_font.mp4" type="video/mp4"></video>
</div>

### Customize bar properties

Set `bar_kwargs` to a dictionary of keyword arguments forwarded to the matploblib `bar` function to control bar properties.


```python
bcr.bar_chart_race(df, bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})
```

<div class="vid">
    <video controls ><source src="../videos/basic_bar_kwargs.mp4" type="video/mp4"></video>
</div>

## Additional features

There are several more additional features to customize the animation.

### Formatting the period

Format the label of the period by setting `period_fmt` to a string with either a date directive or a new-style formatted string.

```python
bcr.bar_chart_race(df, period_fmt='%b %-d, %Y')
```

<div class="vid">
    <video controls ><source src="../videos/other_date_directive.mp4" type="video/mp4"></video>
</div>


### Use numbers for the index instead of dates

It's not necessary to have dates or times in the index of the DataFrame. Below, the index is dropped, which replaces it with integers beginning at 0. These are then interpolated and formatted.

```python
bcr.bar_chart_race(df.reset_index(drop=True), interpolate_period=True, 
                   period_fmt='Index value - {x:.2f}')
```

<div class="vid">
    <video controls ><source src="../videos/other_string_fmt.mp4" type="video/mp4"></video>
</div>

### Add text summarizing the entire period

Define a function that accepts two arguments, the values and ranks of the current period of data, and returns
a dictionary that will be passed to the matplotlib `text` function.

```python
def summary(values, ranks):
    total_deaths = int(round(values.sum(), -2))
    s = f'Total Deaths - {total_deaths:,.0f}'
    return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}

bcr.bar_chart_race(df, period_summary_func=summary)
```

<div class="vid">
    <video controls ><source src="../videos/other_summary.mp4" type="video/mp4"></video>
</div>

## Add a perpendicular bar

Add a single bar perpendicular to the main bars by defining a function that accepts two arguments, the values and ranks of the current period of data, and returns a single number, the position of the bar. You can use string names of aggregation functions that pandas understands.

```python
bcr.bar_chart_race(df, perpendicular_bar_func='mean')
```

<div class="vid">
    <video controls ><source src="../videos/other_perpendicular.mp4" type="video/mp4"></video>
</div>

An example with a user-defined function:

```python
def func(values, ranks):
    return values.quantile(.9)
bcr.bar_chart_race(df, perpendicular_bar_func=func)
```

<div class="vid">
    <video controls ><source src="../videos/other_perpendicular_func.mp4" type="video/mp4"></video>
</div>

## Bar colors

By default, the `'dark24'` colormap is used. This is a qualitative color map, originally found from the [plotly express documentation](https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express). All [matplotlib](https://matplotlib.org/tutorials/colors/colormaps.html) and [plotly](https://plotly.com/python/builtin-colorscales/) colormaps are available by name.

```python
bcr.bar_chart_race(df, cmap='antique')
```

<div class="vid">
    <video controls ><source src="../videos/color_map.mp4" type="video/mp4"></video>
</div>

### Reduce color repetition

It is possible that some colors repeat in your animation, even if there are more colors in the colormap than bars in the animation. This will only happen if you set the `n_bars` parameter, as colors are assigned to each column upon. You'll get a warning advising you to set 
`filter_column_colors` to `True`, which will only assign colors to those bars appearing in the animation.

The following example uses the Accent colormap which has 8 unique colors. The animation is set to have a maximum of 7 bars, but there are still repeating colors.

```python
bcr.bar_chart_race(df, cmap='accent', n_bars=7)
```

!!! warning "`UserWarning`"
    Some of your columns never make an appearance in the animation. To reduce color repetition, set `filter_column_colors` to `True`


<div class="vid">
    <video controls ><source src="../videos/color_warning.mp4" type="video/mp4"></video>
</div>

Setting `filter_column_colors` to `True` will reduce the likelihood of repeating colors, but will still happen if the total number of unique bars is more than the number of colors in the colormap.

```python
bcr.bar_chart_race(df, cmap='accent', n_bars=7, filter_column_colors=True)
```
<div class="vid">
    <video controls ><source src="../videos/color_warning_fixed.mp4" type="video/mp4"></video>
</div>

## Saving the animation

By default, a (very long) string of HTML will be returned from the call to `bar_chart_race`. In order to save the file to disk, use a string of the file name of where you'd like to save. You'll need to [install ffmpeg](../installation#installing-ffmpeg) first in order to save the animation. Once installed, you'll be able to save the animation as a wide variety of formats (mp4, m4v, mov, etc...). To save the animation as a gif, install ImageMagick.

```
bcr.bar_chart_race(df, 'docs/videos/covid19.mp4', figsize=(5, 3))
```
