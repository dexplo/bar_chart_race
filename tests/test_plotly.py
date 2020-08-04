import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bar_chart_race as bcr
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from bar_chart_race import load_dataset, bar_chart_race_plotly


df = load_dataset('covid19')
df = df.iloc[-20:-10]
df1 = df.reset_index(drop=True)


def test_all():
    bar_chart_race_plotly(df)

    df.bcr.bar_chart_race_plotly()

    bar_chart_race_plotly(df, sort='asc')

    bar_chart_race_plotly(df, orientation='v', slider=False)

    bar_chart_race_plotly(df, orientation='v', sort='asc', slider=False)

    bar_chart_race_plotly(df, sort='desc', n_bars=8)

    bar_chart_race_plotly(df, orientation='v', sort='desc', n_bars=8)

    bar_chart_race_plotly(df, sort='asc', n_bars=8, fixed_order=True)

    bar_chart_race_plotly(df, fixed_order=['Iran', 'USA', 'Italy', 'Spain'], period_label={'x': .95, 'y': .9})

    bar_chart_race_plotly(df, fixed_max=True)

    bar_chart_race_plotly(df, fixed_max=True, orientation='v')

    bar_chart_race_plotly(df, sort='asc', steps_per_period=2)

    bar_chart_race_plotly(df, interpolate_period=True, n_bars=8)

    bar_chart_race_plotly(df, n_bars=8, textposition='inside')

    bar_chart_race_plotly(df, n_bars=8, bar_size=.99, layout_kwargs={'height': 800})

    bar_chart_race_plotly(df, n_bars=8, period_label=False)

    bar_chart_race_plotly(df, n_bars=8, sort='asc', orientation='h', period_label={'bgcolor': 'orange', 
                                                                                'font': {'color': 'blue', 'size': 30}})

    bar_chart_race_plotly(df, n_bars=8, period_template='%b %d, %Y')

    bar_chart_race_plotly(df1, n_bars=8, period_template='{x:.1f}', interpolate_period=True)

    bar_chart_race_plotly(df, n_bars=8, interpolate_period=False, bar_textposition='outside', 
                        period_length=500, steps_per_period=10, fixed_max=True)

    def summary(values, ranks):
        total_deaths = int(round(values.sum(), -2))
        s = f'Total Deaths - {total_deaths:,.0f}'
        return {'x': .99, 'y': .05, 'text': s, 'align': 'right', 'size': 8}

    bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary)

    bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary, perpendicular_bar_func='mean')

    bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary, perpendicular_bar_func='max', fixed_max=True)

    def func(values, ranks):
        return values.quantile(.9)

    bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary, perpendicular_bar_func=func)

    bar_chart_race_plotly(df, n_bars=8, period_length=1200)

    bar_chart_race_plotly(df, n_bars=6, sort='asc', colors='Accent')

    bar_chart_race_plotly(df, n_bars=6, sort='asc', colors='Accent', filter_column_colors=True)


    bar_chart_race_plotly(df, n_bars=6, colors=plt.cm.tab20.colors[:19])

    bar_chart_race_plotly(df, colors=['red', 'blue'], filter_column_colors=True)

    bar_chart_race_plotly(df, n_bars=6, title={'text':'Great title', 'font': {'size': 40}, 'x': .5})

    bar_chart_race_plotly(df, n_bars=6, bar_label_font=8, tick_label_font=20)

    bar_chart_race_plotly(df, n_bars=6, bar_label_font={'size': 18, 'family': 'Courier New, monospace', 'color': 'red'})

    bar_chart_race_plotly(df, n_bars=6, scale='log')

    bar_chart_race_plotly(df, 'test.html', n_bars=6, write_html_kwargs={'auto_play': False})
