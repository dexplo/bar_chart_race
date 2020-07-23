import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from ._func_animation import FuncAnimation
from matplotlib.colors import Colormap

from ._common_chart import CommonChart
from ._utils import prepare_wide_data

class _BarChartRace(CommonChart):
    
    def __init__(self, df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                 steps_per_period, period_length, end_period_pause, interpolate_period, 
                 period_label, period_template, period_summary_func, perpendicular_bar_func, 
                 colors, title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                 tick_label_font, tick_template, shared_fontdict, scale, fig, writer, 
                 bar_kwargs, fig_kwargs, filter_column_colors):
        self.filename = filename
        self.extension = self.get_extension()
        self.orientation = orientation
        self.sort = sort
        self.n_bars = n_bars or df.shape[1]
        self.fixed_order = fixed_order
        self.fixed_max = fixed_max
        self.steps_per_period = steps_per_period
        self.period_length = period_length
        self.end_period_pause = end_period_pause
        self.interpolate_period = interpolate_period
        self.period_label = self.get_period_label(period_label)
        self.period_template = period_template
        self.period_summary_func = period_summary_func
        self.perpendicular_bar_func = perpendicular_bar_func
        self.title = self.get_title(title)
        self.bar_size = bar_size
        self.bar_textposition = bar_textposition
        self.bar_texttemplate = bar_texttemplate
        self.bar_label_font = self.get_font(bar_label_font)
        self.tick_label_font = self.get_font(tick_label_font, True)
        self.tick_template = self.get_tick_template(tick_template)
        self.orig_rcParams = self.set_shared_fontdict(shared_fontdict)
        self.scale = scale
        self.fps = 1000 / self.period_length * steps_per_period
        self.writer = self.get_writer(writer)
        self.filter_column_colors = filter_column_colors
        self.extra_pixels = 0
        self.validate_params()

        self.bar_kwargs = self.get_bar_kwargs(bar_kwargs)
        self.html = self.filename is None
        self.df_values, self.df_ranks = self.prepare_data(df)
        self.col_filt = self.get_col_filt()
        self.bar_colors = self.get_bar_colors(colors)
        self.str_index = self.df_values.index.astype('str')
        self.fig_kwargs = self.get_fig_kwargs(fig_kwargs)
        self.subplots_adjust = self.get_subplots_adjust()
        self.fig = self.get_fig(fig)

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

        if self.bar_textposition not in ('outside', 'inside', None):
            raise ValueError('`bar_textposition` must be one of "outside", "inside" or None')

    def get_bar_kwargs(self, bar_kwargs):
        bar_kwargs = bar_kwargs or {}
        if 'width' in bar_kwargs or 'height' in bar_kwargs:
            raise ValueError("Do not set the width or height with `bar_kwargs`. "
                             "Instead, use `bar_size`.")
        if self.orientation == 'h':
            bar_kwargs['height'] = self.bar_size
        else:
            bar_kwargs['width'] = self.bar_size
        if 'alpha' not in bar_kwargs:
            bar_kwargs['alpha'] = .8
        if 'ec' not in bar_kwargs:
            bar_kwargs['ec'] = 'white'
        return bar_kwargs

    def get_period_label(self, period_label):
        if period_label is False:
            return False

        default_period_label = {'size': 12}
        if self.orientation == 'h':
            default_period_label['x'] = .95
            default_period_label['y'] = .15 if self.sort == 'desc' else .85
            default_period_label['ha'] = 'right'
            default_period_label['va'] = 'center'
        else:
            default_period_label['x'] = .95 if self.sort == 'desc' else .05
            default_period_label['y'] = .85
            default_period_label['ha'] = 'right' if self.sort == 'desc' else 'left'
            default_period_label['va'] = 'center'

        if period_label is True:
            return default_period_label
        elif isinstance(period_label, dict):
            period_label = {**default_period_label, **period_label}
        else:
            raise TypeError('`period_label` must be a boolean or dictionary')

        return period_label

    def get_font(self, font, ticks=False):
        default_font_dict = {'size': 7}
        if ticks:
            default_font_dict['ha'] = 'right'
        else:
            if self.orientation == 'h':
                default_font_dict['rotation'] = 0
                default_font_dict['ha'] = 'left'
                default_font_dict['va'] = 'center'
                if self.bar_textposition == 'inside':
                    default_font_dict['ha'] = 'right'
            else:
                default_font_dict['rotation'] = 90
                default_font_dict['ha'] = 'center'
                default_font_dict['va'] = 'bottom'
                if self.bar_textposition == 'inside':
                    default_font_dict['va'] = 'top'

        if font is None:
            font = default_font_dict
        elif isinstance(font, (int, float, str)):
            font = {**default_font_dict, 'size': font}
        elif not isinstance(font, dict):
            raise TypeError('`font` must be a number or dictionary of font properties')
        else:
            font = {**default_font_dict, **font}
        return font

    def prepare_data(self, df):
        if self.fixed_order is True:
            last_values = df.iloc[-1].sort_values(ascending=False)
            cols = last_values.iloc[:self.n_bars].index
            df = df[cols]
        elif isinstance(self.fixed_order, list):
            cols = self.fixed_order
            df = df[cols]
            self.n_bars = min(len(cols), self.n_bars)
            
        compute_ranks = self.fixed_order is False
        dfs = prepare_wide_data(df, self.orientation, self.sort, self.n_bars,
                                self.interpolate_period, self.steps_per_period, compute_ranks)
        if isinstance(dfs, tuple):
            df_values, df_ranks = dfs
        else:
            df_values = dfs

        if self.fixed_order:
            n = df_values.shape[1] + 1
            m = df_values.shape[0]
            rank_row = np.arange(1, n)
            if (self.sort == 'desc' and self.orientation == 'h') or \
                (self.sort == 'asc' and self.orientation == 'v'):
                rank_row = rank_row[::-1]
            
            ranks_arr = np.repeat(rank_row.reshape(1, -1), m, axis=0)
            df_ranks = pd.DataFrame(data=ranks_arr, columns=cols)

        return df_values, df_ranks
        
    def get_col_filt(self):
        col_filt = pd.Series([True] * self.df_values.shape[1])
        if self.n_bars < self.df_ranks.shape[1]:
            orient_sort = self.orientation, self.sort
            if orient_sort in [('h', 'asc'), ('v', 'desc')]:
                # 1 is high
                col_filt = (self.df_ranks < self.n_bars + .99).any()
            else:
                # 1 is low
                col_filt = (self.df_ranks > 0).any()

            if self.filter_column_colors and not col_filt.all():
                self.df_values = self.df_values.loc[:, col_filt]
                self.df_ranks = self.df_ranks.loc[:, col_filt]
        return col_filt
        
    def get_bar_colors(self, colors):
        if colors is None:
            colors = 'dark12'
            if self.df_values.shape[1] > 10:
                colors = 'dark24'

        if isinstance(colors, str):
            from ._colormaps import colormaps
            
            try:
                bar_colors = colormaps[colors.lower()]
            except KeyError:
                raise KeyError(f'Colormap {colors} does not exist. Here are the '
                               f'possible colormaps: {colormaps.keys()}')
        elif isinstance(colors, Colormap):
            bar_colors = colors(range(colors.N)).tolist()
        elif isinstance(colors, list):
            bar_colors = colors
        elif isinstance(colors, tuple):
            bar_colors = list(colors)
        elif hasattr(colors, 'tolist'):
            bar_colors = colors.tolist()
        else:
            raise TypeError('`colors` must be a string name of a colormap, a matplotlib colormap '
                            'instance, list, or tuple of colors')

        # bar_colors is now a list
        n = len(bar_colors)
        orig_bar_colors = bar_colors
        if self.df_values.shape[1] > n:
            bar_colors = bar_colors * (self.df_values.shape[1] // n + 1)
        bar_colors = np.array(bar_colors[:self.df_values.shape[1]])

        if not self.filter_column_colors:
            if not self.col_filt.all():
                col_idx = np.where(self.col_filt)[0] % n
                col_idx_ct = np.bincount(col_idx, minlength=n)
                num_cols = max(self.col_filt.sum(), n)
                exp_ct = np.bincount(np.arange(num_cols) % n, minlength=n)
                if (col_idx_ct > exp_ct).any():
                    warnings.warn("Some of your columns never make an appearance in the animation. "
                                    "To reduce color repetition, set `filter_column_colors` to `True`")
        return bar_colors

    def get_max_plotted_value(self):
        plotted_values = []
        for i in range(len(self.df_values)):
            _, bar_length, _, _ = self.get_bar_info(i)
            plotted_values.append(max(bar_length))
        return max(plotted_values)

    def prepare_axes(self, ax):
        value_axis = ax.xaxis if self.orientation == 'h' else ax.yaxis
        value_axis.grid(True, color='white')
        if self.tick_template:
            value_axis.set_major_formatter(self.tick_template)
        ax.tick_params(labelsize=self.tick_label_font['size'], length=0, pad=2)
        ax.minorticks_off()
        ax.set_axisbelow(True)
        ax.set_facecolor('.9')
        ax.set_title(**self.title)
        # min_val = 1 if self.scale == 'log' else 0

        for spine in ax.spines.values():
            spine.set_visible(False)

        limit = (.2, self.n_bars + .8)
        if self.orientation == 'h':
            ax.set_ylim(limit)
            ax.set_xscale(self.scale)
        else:
            ax.set_xlim(limit)
            ax.set_yscale(self.scale)
            ax.tick_params(axis='x', labelrotation=30)

    def get_subplots_adjust(self):
        import io
        fig = plt.Figure(**self.fig_kwargs)
        ax = fig.add_subplot()
        plot_func = ax.barh if self.orientation == 'h' else ax.bar
        bar_location, bar_length, cols, _ = self.get_bar_info(-1)
        plot_func(bar_location, bar_length, tick_label=cols)
                
        self.prepare_axes(ax)
        texts = self.add_bar_labels(ax, bar_location, bar_length)

        fig.canvas.print_figure(io.BytesIO(), format='png')
        xmin = min(label.get_window_extent().x0 for label in ax.get_yticklabels()) 
        xmin /= (fig.dpi * fig.get_figwidth())
        left = ax.get_position().x0 - xmin + .01

        ymin = min(label.get_window_extent().y0 for label in ax.get_xticklabels()) 
        ymin /= (fig.dpi * fig.get_figheight())
        bottom = ax.get_position().y0 - ymin + .01

        if self.fixed_max:
            if self.orientation == 'h':
                self.fixed_max_value = ax.get_xlim()[1]
            else:
                self.fixed_max_value = ax.get_ylim()[1]

        if self.bar_textposition == 'outside':
            max_bar = max(bar_length)
            if self.orientation == 'h':
                max_bar_pixels = ax.transData.transform((max_bar, 0))[0]
                max_text = max(text.get_window_extent().x1 for text in texts)
            else:
                max_bar_pixels = ax.transData.transform((0, max_bar))[1]
                max_text = max(text.get_window_extent().y1 for text in texts)
            
            self.extra_pixels = max_text - max_bar_pixels + 10

            if self.fixed_max:
                end_pixel = max_bar_pixels + self.extra_pixels
                if self.orientation == 'h':
                    self.fixed_max_value = ax.transData.inverted().transform((end_pixel, 0))[0]
                else:
                    self.fixed_max_value = ax.transData.inverted().transform((0, end_pixel))[1]
        return left, bottom

    def fix_axis_limits(self, ax):
        if self.scale == 'log':
            if self.orientation == 'h':
                ax.set_xlim(1)
            else:
                ax.set_ylim(1)

        if self.fixed_max:
            if self.orientation == 'h':
                ax.set_xlim(None, self.fixed_max_value)
            else:
                ax.set_ylim(None, self.fixed_max_value)

    def create_figure(self):
        fig = plt.Figure(**self.fig_kwargs)
        ax = fig.add_subplot()
        left, bottom = self.subplots_adjust
        fig.subplots_adjust(left=left, bottom=bottom)
        self.prepare_axes(ax)
        self.fix_axis_limits(ax)
        return fig

    def get_bar_info(self, i):
        bar_location = self.df_ranks.iloc[i].values
        top_filt = (bar_location > 0) & (bar_location < self.n_bars + 1)
        bar_location = bar_location[top_filt]
        bar_length = self.df_values.iloc[i].values[top_filt]
        cols = self.df_values.columns[top_filt]
        colors = self.bar_colors[top_filt]
        return bar_location, bar_length, cols, colors

    def set_major_formatter(self, ax):
        if self.tick_template:
            axis = ax.xaxis if self.orientation == 'h' else ax.yaxis
            axis.set_major_formatter(self.tick_template)

    def plot_bars(self, ax, i):
        bar_location, bar_length, cols, colors = self.get_bar_info(i)
        if self.orientation == 'h':
            ax.barh(bar_location, bar_length, tick_label=cols, 
                    color=colors, **self.bar_kwargs)
            ax.set_yticklabels(ax.get_yticklabels(), **self.tick_label_font)
            if not self.fixed_max and self.bar_textposition == 'outside':
                max_bar = bar_length.max()
                new_max_pixels = ax.transData.transform((max_bar, 0))[0] + self.extra_pixels
                new_xmax = ax.transData.inverted().transform((new_max_pixels, 0))[0]
                ax.set_xlim(ax.get_xlim()[0], new_xmax)
        else:
            ax.bar(bar_location, bar_length, tick_label=cols, 
                   color=colors, **self.bar_kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), **self.tick_label_font)
            if not self.fixed_max and self.bar_textposition == 'outside':
                max_bar = bar_length.max()
                new_max_pixels = ax.transData.transform((0, max_bar))[1] + self.extra_pixels
                new_ymax = ax.transData.inverted().transform((0, new_max_pixels))[1]
                ax.set_ylim(ax.get_ylim()[0], new_ymax)

        self.set_major_formatter(ax)
        self.add_period_label(ax, i)
        self.add_period_summary(ax, i)
        self.add_bar_labels(ax, bar_location, bar_length)
        self.add_perpendicular_bar(ax, bar_length, i)

    def add_period_label(self, ax, i):
        if self.period_label:
            if self.period_template:
                idx_val = self.df_values.index[i]
                if self.df_values.index.dtype.kind == 'M':
                    s = idx_val.strftime(self.period_template)
                else:
                    s = self.period_template.format(x=idx_val)
            else:
                s = self.str_index[i]

            if len(ax.texts) == 0:
                # first frame
                ax.text(s=s, transform=ax.transAxes, **self.period_label)
            else:
                ax.texts[0].set_text(s)

    def add_period_summary(self, ax, i):
        if self.period_summary_func:
            values = self.df_values.iloc[i]
            ranks = self.df_ranks.iloc[i]
            text_dict = self.period_summary_func(values, ranks)
            if 'x' not in text_dict or 'y' not in text_dict or 's' not in text_dict:
                name = self.period_summary_func.__name__
                raise ValueError(f'The dictionary returned from `{name}` must contain '
                                  '"x", "y", and "s"')
            ax.text(transform=ax.transAxes, **text_dict)

    def add_bar_labels(self, ax, bar_location, bar_length):
        if self.bar_textposition:
            if self.orientation == 'h':
                zipped = zip(bar_length, bar_location)
            else:
                zipped = zip(bar_location, bar_length)

            delta = .01 if self.bar_textposition == 'outside' else -.01

            text_objs = []
            for x1, y1 in zipped:
                xtext, ytext = ax.transLimits.transform((x1, y1))
                if self.orientation == 'h':
                    xtext += delta
                    val = x1
                else:
                    ytext += delta
                    val = y1

                if callable(self.bar_texttemplate):
                    text = self.bar_texttemplate(val)
                else:
                    text = self.bar_texttemplate.format(x=val)

                xtext, ytext = ax.transLimits.inverted().transform((xtext, ytext))

                text_obj = ax.text(xtext, ytext, text, clip_on=True, **self.bar_label_font)
                text_objs.append(text_obj)
            return text_objs

    def add_perpendicular_bar(self, ax, bar_length, i):
        if self.perpendicular_bar_func:
            if isinstance(self.perpendicular_bar_func, str):
                val = pd.Series(bar_length).agg(self.perpendicular_bar_func)
            else:
                values = self.df_values.iloc[i]
                ranks = self.df_ranks.iloc[i]
                val = self.perpendicular_bar_func(values, ranks)

            if not ax.lines:
                if self.orientation == 'h':
                    ax.axvline(val, lw=10, color='.5', zorder=.5)
                else:
                    ax.axhline(val, lw=10, color='.5', zorder=.5)
            else:
                line = ax.lines[0]
                if self.orientation == 'h':
                    line.set_xdata([val] * 2)
                else:
                    line.set_ydata([val] * 2)
            
    def anim_func(self, i):
        if i is None:
            return
        ax = self.fig.axes[0]
        for bar in ax.containers:
            bar.remove()
        start = int(bool(self.period_label))
        for text in ax.texts[start:]:
            text.remove()
        self.plot_bars(ax, i)
        
    def make_animation(self):
        def init_func():
            ax = self.fig.axes[0]
            self.plot_bars(ax, 0)

        interval = self.period_length / self.steps_per_period
        pause = int(self.end_period_pause // interval)

        def frame_generator(n):
            frames = []
            for i in range(n):
                frames.append(i)
                if pause and i % self.steps_per_period == 0 and i != 0 and i != n - 1:
                    for _ in range(pause):
                        frames.append(None)
            return frames
        
        frames = frame_generator(len(self.df_values))
        anim = FuncAnimation(self.fig, self.anim_func, frames, init_func, interval=interval)

        try:
            fc = self.fig.get_facecolor()
            if fc == (1, 1, 1, 0):
                fc = 'white'
            savefig_kwargs = {'facecolor': fc}
            if self.html:
                ret_val = anim.to_html5_video(savefig_kwargs=savefig_kwargs)
                try:
                    from IPython.display import HTML
                    ret_val = HTML(ret_val)
                except ImportError:
                    pass
            else:
                fc = self.fig.get_facecolor()
                if fc == (1, 1, 1, 0):
                    fc = 'white'
                ret_val = anim.save(self.filename, fps=self.fps, writer=self.writer, 
                                    savefig_kwargs=savefig_kwargs) 
        except Exception as e:
            message = str(e)
            raise Exception(message)
        finally:
            plt.rcParams = self.orig_rcParams

        return ret_val


def bar_chart_race(df, filename=None, orientation='h', sort='desc', n_bars=None, 
                   fixed_order=False, fixed_max=False, steps_per_period=10, 
                   period_length=500, end_period_pause=0, interpolate_period=False, 
                   period_label=True, period_template=None, period_summary_func=None,
                   perpendicular_bar_func=None, colors=None, title=None, bar_size=.95,
                   bar_textposition='outside', bar_texttemplate='{x:,.0f}',
                   bar_label_font=None, tick_label_font=None, tick_template='{x:,.0f}',
                   shared_fontdict=None, scale='linear', fig=None, writer=None, 
                   bar_kwargs=None,  fig_kwargs=None, filter_column_colors=False):
    '''
    Create an animated bar chart race using matplotlib. Data must be in 
    'wide' format where each row represents a single time period and each 
    column represents a distinct category. Optionally, the index can label 
    the time period. Bar length and location change linearly from one 
    time period to the next.

    If no `filename` is given, an HTML string is returned, otherwise the 
    animation is saved to disk.

    You must have ffmpeg installed on your machine to save videos to disk
    and ImageMagick to save animated gifs. Read more here:
    https://www.dexplo.org/bar_chart_race/installation/

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'wide' DataFrame where each row represents a single period 
        of time. Each column contains the values of the bars for that 
        category. Optionally, use the index to label each time period.
        The index can be of any type.

    filename : `None` or str, default None
        If `None` return animation as an HTML5 string. If a string, save 
        animation to that filename location. Use .mp4, .gif, .html, .mpeg, 
        .mov or any other extensions supported by ffmpeg or ImageMagick.

    orientation : 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort : 'desc' or 'asc', default 'desc'
        Sorts the bars. Use 'desc' to place largest bars on top/left 
        and 'asc' to place largest bars on bottom/right.

    n_bars : int, default None
        Choose the maximum number of bars to display on the graph. 
        By default, use all bars. New bars entering the race will appear 
        from the edge of the axes.

    fixed_order : bool or list, default False
        When `False`, bar order changes every time period to correspond 
        with `sort`. When `True`, bars remained fixed according to their 
        final value corresponding with `sort`. Otherwise, provide a list 
        of the exact order of the categories for the entire duration.

    fixed_max : bool, default False
        Whether to fix the maximum value of the axis containing the values.
        When `False`, the axis for the values will have its maximum (xlim/ylim)
        just after the largest bar of the current time period. 
        The axis maximum will change along with the data.

        When True, the maximum axis value will remain constant for the 
        duration of the animation. For example, in a horizontal bar chart, 
        if the largest bar has a value of 100 for the first time period and 
        10,000 for the last time period. The xlim maximum will be 10,000 
        for each frame.

    steps_per_period : int, default 10
        The number of steps to go from one time period to the next. 
        The bars will grow linearly between each period. 
        Increasing this number creates smoother animations.

    period_length : int, default 500
        Number of milliseconds to animate each period (row). 
        Default is 500ms (half of a second).

    end_period_pause : int, default 0
        Number of milliseconds to pause the animation at the end of
        each period. This number must be greater than or equal to 
        period_length / steps_per_period or there will be no pause.
        This is due to all frames having the same time interval.

        By default, each frame is 500 / 10 or 50 milliseconds,
        therefore end_period_pause must be at least 50 for there
        to be a pause. The pause will be in increments of this
        calculated interval and not exact. For example, setting the
        end_period_pause to 725 will produce a pause of 700 
        milliseconds when using the defaults.

    interpolate_period : bool, default `False`
        Whether to interpolate the period. Only valid for datetime or
        numeric indexes. When set to `True`, for example, 
        the two consecutive periods 2020-03-29 and 2020-03-30 with 
        `steps_per_period` set to 4 would yield a new index of
        2020-03-29 00:00:00
        2020-03-29 06:00:00
        2020-03-29 12:00:00
        2020-03-29 18:00:00
        2020-03-30 00:00:00

    period_label : bool or dict, default `True`
        If `True` or dict, use the index as a large text label
        on the axes whose value changes each frame. If `False`,
        don't place label on axes.

        Use a dictionary to supply any valid parameters of the 
        matplotlib `text` method.
        Example:
        {
            'x': .99,
            'y': .8,
            'ha': 'right',
            'va': 'center',
            'size': 8
        }

        The default location depends on `orientation` and `sort`; 
        x and y are in axes units
        * h, desc -> x=.95, y=.15, ha='right', va='center'
        * h, asc -> x=.95, y=.85, ha='right', va='center'
        * v, desc -> x=.95, y=.85, ha='right', va='center'
        * v, asc -> x=.05, y=.85, ha='left', va='center'

    period_template : str, default `None`
        Either a string with date directives or a new-style (Python 3.6+) 
        formatted string. Date directives will only be used for 
        datetime indexes.

        Date directive reference:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        
        Example of string with date directives
            '%B %d, %Y'
        Changes 2020/03/29 to March 29, 2020
        
        For new-style formatted string. Use curly braces and the variable `x`, 
        which will be passed the current period's index value. 
        Example:
            'Period {x:10.2f}'

    period_summary_func : function, default None
        Custom text added to the axes each period.
        Create a user-defined function that accepts one pandas Series of the 
        current time period's values. It must return a dictionary containing 
        the keys "x", "y", and "s" which will be passed to the matplotlib 
        `text` method.
        Example:
        def func(values, ranks):
            total = values.sum()
            s = f'Worldwide deaths: {total}'
            return {'x': .85, 'y': .2, 's': s, 'ha': 'right', 'size': 11}

    perpendicular_bar_func : function or str, default None
        Creates a single bar perpendicular to the main bars that spans the 
        length of the axis. Use either a string that the DataFrame `agg` 
        method understands or a user-defined function.
            
        DataFrame strings - 'mean', 'median', 'max', 'min', etc..

        The function is passed two pandas Series of the current time period's
        data and ranks. It must return a single value.
        Example:
        def func(values, ranks):
            return values.quantile(.75)

    colors : str, matplotlib colormap instance, or list of colors, default 'dark12'
        Colors to be used for the bars. All matplotlib and plotly 
        colormaps are available by string name. Colors will repeat 
        if there are more bars than colors.

        'dark12' is a discrete colormap. If there are more than 12 columns, 
        then the default colormap will be 'dark24'

        Append "_r" to the colormap name to use the reverse of the colormap.
        i.e. "dark12_r"

    title : str or dict, default None
        Title of plot as a string. Use a dictionary to supply several title 
        parameters. You must use the key 'label' for the title.
        Example:
        {
            'label': 'My Bar Chart Race Title',
            'size': 18,
            'color': 'red',
            'loc': 'right',
            'pad': 12
        }

    bar_size : float, default .95
        Height/width of bars for horizontal/vertical bar charts. 
        Use a number between 0 and 1
        Represents the fraction of space that each bar takes up. 
        When equal to 1, no gap remains between the bars.

    bar_textposition : 'outside', 'inside', or None - default 'outside'
        Position where bar label will be placed. Use None when 
        no label is desired.
    
    bar_texttemplate : str or function, default '{x:,.0f}'
        A new-style formatted string to control the formatting
        of the bar labels. Use `x` as the variable name.

        Provide a function that accepts one numeric argument,
        the value of the bar and returns a string
        Example:
        def func(val):
            new_val = int(round(val, -3))
            return f'{new_val:,.0f}'

    bar_label_font : number, str, or dict, default None
        Font size of numeric bar labels. When None, defaults to 7.
        Use a dictionary to supply several font properties.
        Example:
        {
            'size': 12,
            'family': 'Courier New, monospace',
            'color': '#7f7f7f'
        }

    tick_label_font : number or dict, default None
        Font size of tick labels. When None, defaults to 7.
        Use a dictionary to supply several font properties.

    tick_template : str or function, default '{x:,.0f}'
        Formats the ticks on the axis with numeric values 
        (x-axis when horizontal and y-axis when vertical). If given a string,
        pass it to the ticker.StrMethodFormatter matplotlib function. 
        Use 'x' as the variable
        Example: '{x:10.2f}'

        If given a function, its passed to ticker.FuncFormatter, which
        implicitly passes it two variables `x` and `pos` and must return
        a string.

    shared_fontdict : dict, default None
        Dictionary of font properties shared across the tick labels, 
        bar labels, period labels, and title. The only property not shared 
        is `size`. It will be ignored if you try to set it.
        Possible keys are:
            'family', 'weight', 'color', 'style', 'stretch', 'weight', 'variant'
        Example:
        {
            'family' : 'Helvetica',
            'weight' : 'bold',
            'color' : 'rebeccapurple'
        }

    scale : 'linear' or 'log', default 'linear'
        Type of scaling to use for the axis containing the values

    fig : matplotlib Figure, default None
        For greater control over the aesthetics, supply your own figure.

    writer : str or matplotlib Writer instance
        This argument is passed to the matplotlib FuncAnimation.save method.

        By default, the writer will be 'ffmpeg' unless creating a gif,
        then it will be 'imagemagick', or an html file, then it 
        will be 'html'. 
            
        Find all of the availabe Writers:
        >>> from matplotlib import animation
        >>> animation.writers.list()

    bar_kwargs : dict, default None
        Other keyword arguments (within a dictionary) forwarded to the 
        matplotlib `barh`/`bar` function. If no value for 'alpha' is given,
        then it is set to .8 by default.
        Some examples:
            `ec` - edgecolor - color of edge of bar. Default is 'white'
            `lw` - width of edge in points. Default is 1.5
            `alpha` - opacity of bars, 0 to 1

    fig_kwargs : dict, default None
        A dictionary of keyword arguments passed to the matplotlib
        Figure constructor. If not given, figsize is set to (6, 3.5) and 
        dpi to 144.
        Example:
        {
            'figsize': (8, 3),
            'dpi': 120,
            'facecolor': 'red'
        }

    filter_column_colors : bool, default `False`
        When setting n_bars, it's possible that some columns never 
        appear in the animation. Regardless, all columns get assigned
        a color by default. 
        
        For instance, suppose you have 100 columns 
        in your DataFrame, set n_bars to 10, and 15 different columns 
        make at least one appearance in the animation. Even if your 
        colormap has at least 15 colors, it's possible that many 
        bars will be the same color, since each of the 100 columns is
        assigned of the colormaps colors.

        Setting this to `True` will map your colormap to just those 
        columns that make an appearance in the animation, helping
        avoid duplication of colors.

        Setting this to `True` will also have the (possibly unintended)
        consequence of changing the colors of each color every time a 
        new integer for n_bars is used.

        EXPERIMENTAL
        This parameter is experimental and may be changed/removed
        in a later version.

    Returns
    -------
    When `filename` is left as `None`, an HTML5 video is returned as a string.
    Otherwise, a file of the animation is saved and `None` is returned.

    Examples
    --------
    Use the `load_data` function to get an example dataset to 
    create an animation.

    df = bcr.load_dataset('covid19')
    bcr.bar_chart_race(
        df=df, 
        filename='../docs/images/covid19_horiz.gif', 
        orientation='h', 
        sort='desc', 
        n_bars=8, 
        fixed_order=False, 
        fixed_max=True, 
        steps_per_period=20, 
        period_length=500, 
        end_period_pause=0,
        interpolate_period=False, 
        period_label={'x': .98, 'y': .3, 'ha': 'right', 'va': 'center'}, 
        period_template='%B %d, %Y', 
        period_summary_func=lambda v, r: {'x': .98, 'y': .2, 
                                          's': f'Total deaths: {v.sum():,.0f}', 
                                          'ha': 'right', 'size': 11}, 
        perpendicular_bar_func='median', 
        colors='dark12', 
        title='COVID-19 Deaths by Country', 
        bar_size=.95, 
        bar_textposition='inside',
        bar_texttemplate='{x:,.0f}', 
        bar_label_font=7, 
        tick_label_font=7, 
        tick_template='{x:,.0f}',
        shared_fontdict=None, 
        scale='linear', 
        fig=None, 
        writer=None, 
        bar_kwargs={'alpha': .7},
        fig_kwargs={'figsize': (6, 3.5), 'dpi': 144},
        filter_column_colors=False)

    Font Help
    ---------
    Font size can also be a string - 'xx-small', 'x-small', 'small',  
        'medium', 'large', 'x-large', 'xx-large', 'smaller', 'larger'
    These sizes are relative to plt.rcParams['font.size'].
    '''
    bcr = _BarChartRace(df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                        steps_per_period, period_length, end_period_pause, interpolate_period, 
                        period_label, period_template, period_summary_func, perpendicular_bar_func,
                        colors, title, bar_size, bar_textposition, bar_texttemplate, 
                        bar_label_font, tick_label_font, tick_template, shared_fontdict, scale, 
                        fig, writer, bar_kwargs, fig_kwargs, filter_column_colors)
    return bcr.make_animation()
