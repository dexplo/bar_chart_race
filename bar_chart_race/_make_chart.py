import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker, transforms, text, colors

DARK24 = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', 
          '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', 
          '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1', 
          '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']

class _BarChartRace:
    
    def __init__(self, df, filename, orientation, sort, label_bars, use_index, 
                 steps_per_period, period_length, cmap, title, bar_label_size, tick_label_size, 
                 period_label_size, fig, kwargs):
        self.df = df
        self.filename = filename
        self.orientation = orientation
        self.sort = sort
        self.label_bars = label_bars
        self.use_index = use_index
        self.steps_per_period = steps_per_period
        self.period_length = period_length
        self.orig_index = self.df.index.astype('str')
        self.title = title
        self.figsize = (7, 3.5)
        self.dpi = 144
        self.tick_label_size = tick_label_size
        self.bar_label_size = bar_label_size
        self.period_label_size = period_label_size
        self.fps = 1000 / self.period_length * steps_per_period
        self.n_bars = self.df.shape[1]
        self.fig = fig
        self.kwargs = kwargs
        self.validate_params()
        self.html = self.filename is None
        self.bar_colors = self.get_colors(cmap)
        self.x_label, self.y_label = self.get_label_position()
        self.df_values, self.df_rank = self.prepare_data()
        if self.fig is not None:
            self.fig, self.ax = fig, fig.axes[0]
            self.figsize = fig.get_size_inches()
            self.dpi = fig.dpi
        else:
            self.fig, self.ax = self.create_figure()

    def validate_params(self):
        if isinstance(self.filename, str):
            if '.' not in self.filename:
                raise ValueError('`filename` must have an extension')
        elif self.filename is not None:
            raise TypeError('`filename` must be None or a string')
            
        if self.sort not in ('asc', 'desc'):
            raise ValueError('`sort` must be "asc" or "desc"')

        if self.orientation not in ('h', 'v'):
            raise ValueError('`orientation` must be "h" or "v"')

        if self.fig is not None and not isinstance(self.fig, plt.Figure):
            raise TypeError('`fig` must be a matplotlib Figure instance')

    def get_colors(self, cmap):
        if isinstance(cmap, str):
            if cmap == 'dark24':
                cmap = DARK24
            else:
                cmap = plt.cm.get_cmap(cmap)

        if isinstance(cmap, colors.Colormap):
            bar_colors = cmap(range(cmap.N)).tolist()
        elif isinstance(cmap, list):
            bar_colors = cmap
        elif hasattr(cmap, 'tolist'):
            bar_colors = cmap.tolist()
        else:
            raise TypeError('`cmap` must be a string name of a colormap, a matplotlib colormap instance' \
                            'or a list of colors')

        # bar_colors is now a list
        n = len(bar_colors)
        if self.n_bars > n:
            bar_colors = bar_colors * (self.n_bars // n + 1)
        return np.array(bar_colors[:self.n_bars])

    def get_label_position(self):
        if self.orientation == 'h':
            x_label = .6
            y_label = .25 if self.sort == 'desc' else .8
        else:
            x_label = .7 if self.sort == 'desc' else .1
            y_label = .8
        return x_label, y_label
        
    def prepare_data(self):
        df_values = self.df.reset_index(drop=True)
        df_values.index = df_values.index * self.steps_per_period
        df_rank = df_values.rank(axis=1, method='first', ascending=False)
        if (self.sort == 'desc' and self.orientation == 'h') or (self.sort == 'asc' and self.orientation == 'v'):
            df_rank = self.n_bars + 1 - df_rank
        new_index = range((len(df_values) - 1) * self.steps_per_period + 1)
        df_values = df_values.reindex(new_index).interpolate()
        df_rank = df_rank.reindex(new_index).interpolate()
        return df_values, df_rank
        
    def create_figure(self):
        fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)
        limit = (.3, self.n_bars + .8)
        rect = self.calculate_new_figsize(fig)
        ax = fig.add_axes(rect)
        if self.orientation == 'h':
            ax.set_ylim(limit)
            ax.grid(True, axis='x', color='white')
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        else:
            ax.set_xlim(limit)
            ax.grid(True, axis='y', color='white')
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=30)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=self.tick_label_size, pad=2)
        ax.set_facecolor('.9')
        ax.set_title(self.title)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax

    def calculate_new_figsize(self, real_fig):
        import io
        fig = plt.Figure(tight_layout=True, figsize=self.figsize)
        ax = fig.add_subplot()
        fake_cols = [chr(i + 70) for i in range(self.df.shape[1])]
        if self.orientation == 'h':
            ax.barh(fake_cols, [1] * self.df.shape[1])
            ax.tick_params(labelrotation=0, axis='y', labelsize=self.tick_label_size)
            ax.set_title(self.title)
            fig.canvas.print_figure(io.BytesIO())
            orig_pos = ax.get_position()
            ax.set_yticklabels(self.df.columns)
        else:
            ax.bar(fake_cols, [1] * self.df.shape[1])
            ax.tick_params(labelrotation=30, axis='x', labelsize=self.tick_label_size)
            ax.set_title(self.title)
            fig.canvas.print_figure(io.BytesIO())
            orig_pos = ax.get_position()
            ax.set_xticklabels(self.df.columns, ha='right')
        
        fig.canvas.print_figure(io.BytesIO(), format='png')
        new_pos = ax.get_position()

        if self.orientation == 'h':
            coord, prev_coord = new_pos.x0, orig_pos.x0
            old_fig_size = self.figsize[0]
        else:
            coord, prev_coord = new_pos.y0, orig_pos.y0
            old_fig_size = self.figsize[1]

        if coord > prev_coord:
            prev_inches = prev_coord * old_fig_size
            total_inches = coord * old_fig_size
            extra_inches = total_inches - prev_inches
            new_fig_inches = extra_inches + old_fig_size
            if self.orientation == 'h':
                real_fig.set_size_inches(new_fig_inches, self.figsize[1])
                ax_start = total_inches / new_fig_inches
                ax_length = orig_pos.x1 - ax_start
                return [ax_start, orig_pos.y0, ax_length, orig_pos.y1 - orig_pos.y0]
            else:
                real_fig.set_size_inches(self.figsize[0], new_fig_inches)
                ax_start = total_inches / new_fig_inches
                ax_length = orig_pos.y1 - ax_start
                return [orig_pos.x0, ax_start, orig_pos.x1 - orig_pos.x0, ax_length]
        return [orig_pos.x0, orig_pos.y0, orig_pos.x1 - orig_pos.x0, orig_pos.y1 - orig_pos.y0]
            
    def plot_bars(self, i):
        bar_location = self.df_rank.iloc[i]
        nas = bar_location.isna()
        bar_location = bar_location.dropna()
        bar_length = self.df_values.iloc[i].dropna()
        cols = self.df.columns[~nas]
        if self.orientation == 'h':
            self.ax.barh(bar_location, bar_length, ec='white', tick_label=cols, color=self.bar_colors[~nas], **self.kwargs)
            self.ax.set_xlim(self.ax.get_xlim()[0], bar_length.max() * 1.1)
        else:
            self.ax.bar(bar_location, bar_length, ec='white', tick_label=cols, color=self.bar_colors[~nas], **self.kwargs)
            self.ax.set_ylim(self.ax.get_ylim()[0], bar_length.max() * 1.16)
            
        t = self.ax.texts
        num_texts = len(t)
        if self.use_index:
            val = self.orig_index[i // self.steps_per_period]
            if num_texts == 0:
                self.ax.text(self.x_label, self.y_label, val, transform=self.ax.transAxes, fontsize=self.period_label_size)
            else:
                t[0].set_text(val)

        if self.label_bars:
            if self.orientation == 'h':
                zipped = zip(bar_length, bar_location)
            else:
                zipped = zip(bar_location, bar_length)

            for j, (x1, y1) in enumerate(zipped, start=int(self.use_index)):
                xtext, ytext = self.ax.transLimits.transform((x1, y1))
                if self.orientation == 'h':
                    xtext += .01
                    text = f'{x1:,.0f}'
                    rotation = 0
                    ha = 'left'
                    va = 'center'
                else:
                    ytext += .015
                    text = f'{y1:,.0f}'
                    rotation = 90
                    ha = 'center'
                    va = 'bottom'
                xtext, ytext = self.ax.transLimits.inverted().transform((xtext, ytext))
                if j >= num_texts:
                    self.ax.text(xtext, ytext, text, ha=ha, rotation=rotation,
                                 fontsize=self.bar_label_size, va=va)
                else:
                    t[j].set_text(text)
                    t[j].set_position((xtext, ytext))

    def anim_func(self, i):
        for bar in self.ax.containers:
            bar.remove()
        self.plot_bars(i)
        
    def make_animation(self):
        def init_func():
            self.plot_bars(0)
        
        interval = self.period_length / self.steps_per_period
        anim = FuncAnimation(self.fig, self.anim_func, range(1, len(self.df_values)), 
                             init_func, interval=interval)

        if self.html:
            return anim.to_html5_video()

        extension = self.filename.split('.')[-1]
        if extension == 'gif':
            anim.save(self.filename, fps=self.fps, writer='imagemagick')
        else:
            anim.save(self.filename, fps=self.fps)


def bar_chart_race(df, filename=None, orientation='h', sort='desc', label_bars=True, 
                   use_index=True, steps_per_period=10, period_length=500, cmap='dark24', 
                   title=None, bar_label_size=7, tick_label_size=7, period_label_size=16, 
                   fig=None, **kwargs):
    '''
    Create an animated bar chart race using matplotlib. Data must be in 'wide' format where each
    row represents a single time period and each column represents a distinct category. 
    Optionally, the index can label the time period.

    Bar height and location change linearly from one time period to the next.

    This is resource intensive - Start with just a few rows of data

    Parameters
    ----------
    df : pandas DataFrame
        Must be 'wide' where each row represents a single period of time. Each column contains
        the values of the bars for that category. Optionally, use the index to label each time period.

    filename : `None` or str, default None
        If `None` return animation as HTML5.
        If a string, save animation to that filename location. Use .mp4 or .gif extensions

    orientation: 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort: 'desc' or 'asc', default 'desc'
        Choose how to sort the bars. Use 'desc' to put largest bars on top and 'asc' to place largest
        bars on bottom.
    
    label_bars : bool, default `True`
        Whether to label the bars with their value on their right

    use_index : bool, default `True`
        Whether to use the index as the text in the plot

    steps_per_period : int, default 10
        The number of steps to go from one time period to the next. 
        The bar will grow linearly between each period.

    period_length : int, default 500
        Number of milliseconds to animate each period (row). Default is 500ms (half of a second)

    cmap : str, matplotlib colormap instance, or list of colors, default 'dark24'
        Colors to be used for the bars. Colors will repeat if there are more bars
        than colors.

    title: str, default None
        Title of plot

    bar_label_size: int, float, default 7
        Size in points of numeric labels just outside of the bars

    tick_label_size: int, float, default 7
        Size in points of tick labels

    period_label_size: int, float, default 16
        Size in points of label plotted with the axes that labels the period.

    fig : matplotlib Figure, default None
        For greater control over the aesthetics, supply your own figure with a single axes.

    **kwargs : key, value pairs
        Other keyword arguments passed to the matplotlib barh/bar function.


    Returns
    -------
    Either HTML5 video or creates an mp4/gif file of the animation and returns `None`

    Notes
    -----
    Default figure size is 7 by 3.5 with 144 DPI

    It is possible for some bars to be out of order momentarily during a transition since
    both height and location change linearly.

    Examples
    --------
    Use the `load_data` function to get an example dataset to create an animation.

    df = bcr.load_data('covid19')
    bcr.bar_chart_race(
        df=df,
        filename='covid19_horiz_desc.mp4',
        orientation='h',
        sort='desc',
        label_bars=True,
        use_index=True,
        steps_per_period=10,
        period_length=500,
        cmap='dark24',
        title='COVID-19 Deaths by Country',
        bar_label_size=7,
        tick_label_size=7,
        period_label_size=16,
        fig=None)
    '''
    bcr = _BarChartRace(df, filename, orientation, sort, label_bars, use_index, steps_per_period, 
                        period_length, cmap, title, bar_label_size, tick_label_size, 
                        period_label_size, fig, kwargs)
    return bcr.make_animation()

def load_dataset(name='covid19'):
    '''
    Return a pandas DataFrame suitable for immediate use in `bar_chart_race`

    Parameters
    ----------
    name : str, default 'covid19'
        Name of dataset to load. Currently, only one is available.

    Returns
    -------
    pandas DataFrame
    '''
    return pd.read_csv(f'https://raw.githubusercontent.com/dexplo/bar_chart_race/master/data/{name}.csv', 
                       index_col='date', parse_dates=['date'])