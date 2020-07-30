import pytest
import pandas as pd
import matplotlib.pyplot as plt
import bar_chart_race as bcr


df_race = pd.read_csv('tests/data/lcr_data.csv', index_col='date', parse_dates=['date'])
s = pd.Series({'US': 330, 'United Kingdom': 65, 'Brazil': 220, 'Italy': 60, 'France': 126})
df_pop = df_race[s.index] / s


class TestBasics:

    def test_default(self):
        bcr.line_chart_race(df_race.iloc[:, -5:])

    def test_lines(self):
        bcr.line_chart_race(df_race, n_lines=8)

    def test_images(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country')

    def test_others_line_func(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=True)

    def test_steps_per_period(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=True, steps_per_period=20)
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=True, steps_per_period=3)

    def test_period_length(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=True, 
                    period_length=200, steps_per_period=30)

    def test_others_line_func_agg(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func='mean', steps_per_period=5)
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func='mean', 
                            others_line_kwargs={'s': 'Mean Others', 'color':'.5', 'lw': 3}, steps_per_period=5)
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func='mean', steps_per_period=5)

    def test_others_line_func_udf(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=lambda x: x.sum(), 
                    others_line_kwargs={'s': 'Sum Others', 'color':'.5', 'lw': 3}, steps_per_period=5)

    def test_agg_line_func(self):
        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=lambda x: x.sum(), 
                            period_length=1000, others_line_kwargs={'s': 'Sum Others', 'lw': 3, 'ls': '--'}, 
                            agg_line_func='median', agg_line_kwargs={'s': 'Median All'},
                            steps_per_period=5)

        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=lambda x: x.sum(), 
                            period_length=1000, others_line_kwargs={'s': 'Sum Others', 'lw': 3, 'ls': '--'}, 
                            agg_line_func='sum', agg_line_kwargs={'s': 'Sum All'},
                            steps_per_period=5)

        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=lambda x: x.sum(), 
                            period_length=1000, end_period_pause=1000, 
                            others_line_kwargs={'s': 'Sum Others', 'lw': 3, 'ls': '--'}, 
                            agg_line_func='median', agg_line_kwargs={'s': 'Median Shown'},
                            steps_per_period=5)

    def test_period_summary_func(self):
        def psf(values):
                total = values.sum()
                s = f'Worldwide Deaths: {total:,.0f}'
                return {'x': .05, 'y': .85, 's': s, 'size': 10}

        bcr.line_chart_race(df_race, n_lines=4, images='country', others_line_func=lambda x: x.sum(),  
                            others_line_kwargs={'s': 'Sum Others', 'lw': 3, 'ls': '--'}, 
                            agg_line_func='sum', agg_line_kwargs={'s': 'Sum All'},
                            steps_per_period=5, period_summary_func=psf)

    def test_line_width_data(self):
        bcr.line_chart_race(df_race[df_pop.columns], n_lines=5, images='country',
                            steps_per_period=5, line_width_data=df_pop)

    def test_fade(self):
        bcr.line_chart_race(df_race[df_pop.columns],  images='country',
                            steps_per_period=5, line_width_data=df_pop, fade=.9)

        bcr.line_chart_race(df_race[df_pop.columns], n_lines=5, images='country',
                            steps_per_period=5, line_width_data=df_pop, fade=.8, min_fade=0)

    def test_images(self):
        url = 'https://icons.iconarchive.com/icons/wikipedia/flags/1024/US-United-States-Flag-icon.png'
        images = [url] * 5
        bcr.line_chart_race(df_race, n_lines=5, images=images, title='COVID-19 Deaths',
                            steps_per_period=5, line_width_data=df_pop, fade=.9)

    def test_colors(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, colors='tab20')

        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, colors=plt.cm.Accent)

    def test_font(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, line_label_font=5,
                           tick_label_font=4)

        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, 
                            line_label_font={'size': 9, 'color': 'red'}, tick_label_font=4)

    def test_tick_template(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, 
                            tick_template=lambda x, pos: f'{x / 1000:.0f}k')

        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, 
                            tick_template='deaths {x:.2f}')
    
    def test_scale(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, scale='linear')

    def test_fig(self):
        fig = plt.Figure(figsize=(6, 3), facecolor='tan', dpi=120)
        fig.add_subplot(1, 2, 1)
        fig.add_subplot(1, 2, 2)
        bcr.line_chart_race(df_race, n_lines=5, images='country', steps_per_period=5, fig=fig)

    def test_line_kwargs(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', 
                            steps_per_period=5, line_kwargs={"ls": '--', 'lw': 2, 'alpha': .3})

    def test_fig_kwargs(self):
        bcr.line_chart_race(df_race, n_lines=5, images='country', 
                            steps_per_period=5, fig_kwargs={'figsize': (4, 2), 'dpi': 100, 'facecolor': 'yellow'})

    def test_videos(self):
        bcr.line_chart_race(df_race, 'tests/videos/lcr_yellow.mp4', n_lines=5, images='country', 
                            steps_per_period=5, fig_kwargs={'facecolor': 'yellow'})

        bcr.line_chart_race(df_race, 'tests/videos/test_html.html', n_lines=5, images='country', 
                            steps_per_period=5)
