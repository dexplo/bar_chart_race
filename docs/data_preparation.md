# Data Preparation

bar_chart_race exposes two functions, `prepare_wide_data` and `prepare_long_data` to transform pandas DataFrames to the correct form.

## Wide data

 To show how the `prepare_wide_data` function works, we'll read in the last three rows from the `covid19_tutorial` dataset.

```python
df = bcr.load_dataset('covid19_tutorial').tail(3)
df
```

{% include 'html/data_preparation_1.html' %}

This format of data is sometimes known as 'wide' data since each column contains data that all represents the same thing (deaths). Each new country would add an additional column to the DataFrame, making it wider. This is the type of data that the `bar_chart_race` function requires.

The `prepare_wide_data` function is what `bar_chart_race` calls internally, so it isn't necessary to use directly. However, it is available so that you can view and understand how the data gets prepared. To transition the bars smoothly from one time period to the next, both the length of the bars and position are changed linearly. Two DataFrames of the same shape are returned - one for the values and the other for the ranks.

```python
df_values, df_ranks = bcr.prepare_wide_data(df, steps_per_period=4, 
                                            orientation='h', sort='desc')
```

Below, we have the `df_values` DataFrame containing the length of each bar for each frame. A total of four rows now exist for each period.

{% include 'html/data_preparation_2.html' %}

The `df_ranks` DataFrame contains the numerical ranking of each country and is used for the position of the bar along the y-axis (or x-axis when veritcal). Notice that there are two sets of bars that switch places.

{% include 'html/data_preparation_3.html' %}

### Don't use before animation

There is no need to use this function before making the animation if you already have wide data. Pass the `bar_chart_race` function your original data.

## Long data

'Long' data is a format for data where all values of the same kind are stored in a single column. Take a look at the baseball data below, which contains the cumulative number of home runs each of the top 20 home run hitters accumulated by year.

```python
df_baseball = bcr.load_dataset('baseball')
df_baseball
```

{% include 'html/data_preparation_4.html' %}

Name, year, and home runs are each in a single column, contrasting with the wide data, where each column had the same type of data. Long data must be converted to wide data by pivoting categorical column and placing the period in the index. The `prepare_long_data` provides this functionality. It simply uses the pandas `pivot_table` method to pivot (and potentially aggregate) the data before passing it to `prepare_wide_data`. The same two DataFrames are returned.

```python
df_values, df_ranks = bcr.prepare_long_data(df_baseball, index='year', columns='name',
                                            values='hr', steps_per_period=5)
df_values.head(16)
```

The linearly interpolated values for the first three seasons of each player:

{% include 'html/data_preparation_5.html' %}

The rankings change substantially during this time period.

```python
df_ranks.head(16)
```

{% include 'html/data_preparation_6.html' %}

### Usage before animation

If you wish to use this function before an animation, set `steps_per_period` to 1.

```python
df_values, df_ranks = bcr.prepare_long_data(df_baseball, index='year', columns='name',
                                            values='hr', steps_per_period=1,
                                            orientation='h', sort='desc')

def period_summary(values, ranks):
    top2 = values.nlargest(2)
    leader = top2.index[0]
    lead = top2.iloc[0] - top2.iloc[1]
    s = f'{leader} by {lead:.0f}'
    return {'s': s, 'x': .95, 'y': .07, 'ha': 'right', 'size': 8}

bcr.bar_chart_race(df_values, period_length=1000,
                   fixed_max=True, fixed_order=True, n_bars=10,
                   figsize=(5, 3), period_fmt='Season {x:,.0f}',
                   title='Top 10 Home Run Hitters by Season Played')
```

{% macro video(name) %}
    <div class="vid">
        <video controls ><source src="../videos/{{ name }}.mp4" type="video/mp4"></video>
    </div>
{% endmacro %}

<div>{{ video('prepare_long') }}</div>