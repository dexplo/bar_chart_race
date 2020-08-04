import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ._func_animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import ticker, colors as mcolors, dates as mdates
from matplotlib import image as mimage
from matplotlib import patches as mpatches

from ._common_chart import CommonChart
from ._utils import prepare_wide_data


OTHERS_COLOR = .7, .7, .7, .6
AGG_COLOR = 0, 0, 0, 1


class _LineChartRace(CommonChart):
    
    def __init__(self, df, filename, n_lines, steps_per_period, period_length, 
                 end_period_pause, period_summary_func, line_width_data, agg_line_func, 
                 agg_line_kwargs, others_line_func, others_line_kwargs, fade, min_fade, 
                 images, colors, title, line_label_font, tick_label_font, tick_template, 
                 shared_fontdict, scale, fig, writer, line_kwargs, fig_kwargs):
        self.filename = filename
        self.extension = self.get_extension()
        self.n_lines = n_lines or df.shape[1]
        self.steps_per_period = steps_per_period
        self.period_length = period_length
        self.end_period_pause = end_period_pause
        self.period_summary_func = period_summary_func
        self.agg_line_func = agg_line_func
        self.agg_line_kwargs, self.agg_line_label = self.get_line_kwargs(agg_line_kwargs, 'agg')
        self.others_line_func = others_line_func
        self.others_line_kwargs, self.others_line_label = self.get_line_kwargs(others_line_kwargs, 'others')
        self.line_width_data = self.get_line_width_data(line_width_data)
        self.fade = fade
        self.min_fade = min_fade
        self.title = self.get_title(title)
        self.line_label_font = self.get_font(line_label_font)
        self.tick_label_font = self.get_font(tick_label_font, True)
        self.tick_template = self.get_tick_template(tick_template)
        self.orig_rcParams = self.set_shared_fontdict(shared_fontdict)
        self.scale = scale
        self.writer = self.get_writer(writer)
        self.fps = 1000 / self.period_length * steps_per_period
        self.validate_params()

        self.line_kwargs = self.get_line_kwargs(line_kwargs, 'line')
        self.html = self.filename is None
        self.all_values, self.df_values, self.df_ranks, self.df_others, self.others_agg_line = self.prepare_data(df)
        self.agg_line = self.prepare_agg_line()
        self.is_x_date = self.df_values.index.dtype.kind == 'M'
        self.colors = self.get_colors(colors)
        self.str_index = self.df_values.index.astype('str')
        self.fig_kwargs = self.get_fig_kwargs(fig_kwargs)
        self.subplots_adjust = self.get_subplots_adjust()
        self.fig = self.get_fig(fig)
        self.collections = {}
        self.other_collections = {}
        self.texts = {}
        self.images = self.get_images(images)
        self.image_radius = self.fig.get_figwidth() * self.fig.dpi * .02

    def get_line_kwargs(self, kwargs, kind):
        defaults = {'lw': 1.5, 'ls': '-'}
        text = None
        if kind == 'others':
            defaults['color'] = OTHERS_COLOR
        elif kind == 'agg':
            defaults['color'] = AGG_COLOR
        elif kind == 'line':
            defaults['alpha'] = 1

        if kwargs is None:
            kwargs = defaults
        elif isinstance(kwargs, dict):
            kwargs = {**defaults, **kwargs}
            if 's' in kwargs:
                text = kwargs.pop('s')
        else:
            raise TypeError(f'{kind}_line_kwargs must be a dictionary with line properties')
        
        if 'linewidth' in kwargs:
            kwargs['lw'] = kwargs.pop('linewidth')
        if 'linestyle' in kwargs:
            kwargs['ls'] = kwargs.pop('linestyle')
        if kind == 'line':
            return kwargs
        if 'c' in kwargs:
            kwargs['color'] = kwargs.pop('c')
        
        kwargs['color'] = mcolors.to_rgba(kwargs['color'])
        return kwargs, text

    def get_line_width_data(self, line_width_data):
        if line_width_data is None:
            return
        df = line_width_data.copy()
        min_val = df.min().min()
        df = df - min_val
        max_val = df.max().max()
        df = df / max_val
        df = df * 6 + 1
        return df

    def validate_params(self):
        if isinstance(self.filename, str):
            if '.' not in self.filename:
                raise ValueError('`filename` must have an extension')
        elif self.filename is not None:
            raise TypeError('`filename` must be None or a string')

    def get_font(self, font, ticks=False):
        default_font_dict = {'size': 7, 'ha': 'left', 'va': 'center'}
        if ticks:
            default_font_dict['ha'] = 'center'

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
        sort = 'desc'
        interpolate_period = True
        compute_ranks = sort = True
        orientation = 'h'
        values, ranks =  prepare_wide_data(df, orientation, sort, self.n_lines, interpolate_period, 
                                           self.steps_per_period, compute_ranks)

        idx = values.iloc[-1].sort_values(ascending=False).index
        top_cols, other_cols = idx[:self.n_lines], idx[self.n_lines:]
        all_values = values.copy()
        values, ranks, others = values[top_cols], ranks[top_cols], values[other_cols]

        if self.others_line_func in (None, True) or len(other_cols) == 0:
            others_agg_line = None
        else:
            others_agg_line = self.prepare_others_agg_line(others)
        return all_values, values, ranks, others, others_agg_line

    def prepare_others_agg_line(self, others):
        if isinstance(self.others_line_func, str):
            s_others = others.agg(self.others_line_func, axis=1)
            label = self.others_line_func
        elif callable(self.others_line_func):
            s_others = others.agg(self.others_line_func, axis=1)
            label = self.others_line_func.__name__
        else:
            raise TypeError('`others_line_func` must be either a string or function')

        if self.others_line_label is None:
            self.others_line_label = label
        return s_others

    def prepare_agg_line(self):
        if self.agg_line_func is None:
            return
        if isinstance(self.agg_line_func, str):
            s_agg = self.all_values.agg(self.agg_line_func, axis=1)
            label = self.agg_line_func
        elif callable(self.agg_line_func):
            s_agg = self.all_values.agg(self.agg_line_func, axis=1)
            label = self.agg_line_func.__name__
        else:
            raise TypeError('`agg_line_func` must be either a string or function')

        if self.agg_line_label is None:
            self.agg_line_label = label
        return s_agg

    def get_colors(self, colors):
        if colors is None:
            colors = 'dark12'
            if self.df_values.shape[1] > 10:
                colors = 'dark24'
            
        if isinstance(colors, str):
            from ._colormaps import colormaps
            try:
                colors = colormaps[colors.lower()]
            except KeyError:
                raise KeyError(f'Colormap {colors} does not exist. Here are the '
                               f'possible colormaps: {colormaps.keys()}')
        elif isinstance(colors, mcolors.Colormap):
            colors = colors(range(colors.N)).tolist()
        elif isinstance(colors, list):
            colors = colors
        elif isinstance(colors, tuple):
            colors = list(colors)
        elif hasattr(colors, 'tolist'):
            colors = colors.tolist()
        else:
            raise TypeError('`colors` must be a string name of a colormap, a matplotlib colormap '
                            'instance, list, or tuple of colors')

        # colors is a list
        n = len(colors)
        if self.df_values.shape[1] > n:
            colors = colors * (self.df_values.shape[1] // n + 1)

        colors = mcolors.to_rgba_array(colors)
        colors = colors[:self.df_values.shape[1]]
        return dict(zip(self.df_values.columns, colors))

    def prepare_axes(self, ax):
        ax.grid(True, color='white')
        ax.tick_params(labelsize=self.tick_label_font['size'], length=0, pad=2)
        ax.minorticks_off()
        ax.set_axisbelow(True)
        ax.set_facecolor('.9')
        ax.set_title(**self.title)
        
        min_val = self.df_values.min().min()
        max_val = self.df_values.max().max()

        if self.others_line_func is True:
            min_val = min(min_val, self.df_others.min().min())
            max_val = max(max_val, self.df_others.max().max())
        elif self.others_agg_line is not None:
            min_val = min(min_val, self.others_agg_line.min())
            max_val = max(max_val, self.others_agg_line.max())
            
        if self.agg_line is not None:
            min_val = min(min_val, self.agg_line.min())
            max_val = max(max_val, self.agg_line.max())
        min_val = 1 if self.scale == 'log' else min_val

        ax.set_yscale(self.scale)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlim(self.df_values.index[0], self.df_values.index[-1])
        ax.set_ylim(min_val, max_val)
        xmin, xmax = ax.get_xlim()
        delta = (xmax - xmin) * .05
        ax.set_xlim(xmin - delta, xmax + delta)

        ymin, ymax = ax.get_ylim()
        delta = (ymax - ymin) * .05
        if self.scale == 'log':
            delta = 0
            ymax = 2 * ymax
        ax.set_ylim(ymin - delta, ymax + delta)

        if self.is_x_date:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d'))

        if self.tick_template:
            ax.yaxis.set_major_formatter(self.tick_template)

    def get_subplots_adjust(self):
        import io
        fig = plt.Figure(**self.fig_kwargs)
        ax = fig.add_subplot()
        ax.plot(self.df_values)
                
        self.prepare_axes(ax)
        fig.canvas.print_figure(io.BytesIO(), format='png')
        xmin = min(label.get_window_extent().x0 for label in ax.get_yticklabels()) 
        xmin /= (fig.dpi * fig.get_figwidth())
        left = ax.get_position().x0 - xmin + .01

        ymin = min(label.get_window_extent().y0 for label in ax.get_xticklabels()) 
        ymin /= (fig.dpi * fig.get_figheight())
        bottom = ax.get_position().y0 - ymin + .01
        return left, bottom

    def get_fig(self, fig):
        if fig is not None and not isinstance(fig, plt.Figure):
            raise TypeError('`fig` must be a matplotlib Figure instance')
        if fig is not None:
            if not fig.axes:
                raise ValueError('The figure passed to `fig` must have an axes')
            ax = fig.axes[0]
            self.prepare_axes(ax)
        else:
            fig = self.create_figure()
        return fig

    def create_figure(self):
        fig = plt.Figure(**self.fig_kwargs)
        ax = fig.add_subplot()
        left, bottom = self.subplots_adjust
        fig.subplots_adjust(left=left, bottom=bottom)
        self.prepare_axes(ax)
        return fig

    def get_images(self, images):
        if images is None:
            return
        if isinstance(images, str):
            from ._utils import read_images
            return read_images(images.lower(), self.df_values.columns)
        if len(images) != len(self.df_values.columns):
            raise ValueError('The number of images does not match the number of columns')
        if isinstance(images, list):
            images = dict(zip(self.df_values.columns, images))
        return {col: mimage.imread(image) for col, image in images.items()}
            
    def get_visible(self, i):
        n = 1_000_000 # make all visible until better logic here
        return (self.df_ranks.iloc[i] <= self.n_lines + n + .5).to_dict()

    def add_period_summary(self, ax, s):
        if self.period_summary_func:
            text_dict = self.period_summary_func(s)
            if 'x' not in text_dict or 'y' not in text_dict or 's' not in text_dict:
                name = self.period_summary_func.__name__
                raise ValueError(f'The dictionary returned from `{name}` must contain '
                                  '"x", "y", and "s"')
            return text_dict

    def anim_func(self, i):
        if i is None:
            return
        ax = self.fig.axes[0]
        s = self.df_values.iloc[i]
        s_all = self.all_values.iloc[i]
        x, y = s.name, s.to_dict()
        if self.is_x_date:
            x = mdates.date2num(x)

        if self.images:
            xmin, xmax = ax.get_xlim()
            x_extra = (xmax - xmin) * .025
        else:
            x_extra = 0

        visible = self.get_visible(i)

        if self.agg_line is not None:
            y['___agg_line___'] = self.agg_line.iloc[i]
            visible['___agg_line___'] = True

        if self.others_agg_line is not None:
            y['___others_line___'] = self.others_agg_line.iloc[i]
            visible['___others_line___'] = True

        for col, collection in self.collections.items():
            text = self.texts[col]
            val = y[col]
            color = self.colors[col]
            vis = visible[col]

            seg = collection.get_segments()
            last = seg[-1][-1]
            new_seg = np.row_stack((last, [x, val]))
            seg.append(new_seg)
            collection.set_segments(seg)
            color_arr = collection.get_colors()

            color_arr = np.append(color_arr, [color], axis=0)
            color_arr[:, -1] = np.clip(color_arr[:, -1] * self.fade, self.min_fade, None)
            collection.set_color(color_arr)

            is_other_agg = col in ('___others_line___', '___agg_line___')
            if self.line_width_data is not None and not is_other_agg:
                lw = self.line_width_data.iloc[i // self.steps_per_period][col]
                lw_arr = collection.get_linewidths()
                lw_arr = np.append(lw_arr, [lw], axis=0)
                collection.set_linewidths(lw_arr)

            text.set_position((x + x_extra, val))
            text.set_visible(vis)
            collection.set_visible(vis)

        if self.others_line_func is True:
            y_other = self.df_others.iloc[i].to_dict()
            for col, collection in self.other_collections.items():
                val = y_other[col]
                seg = collection.get_segments()
                last = seg[-1][-1]
                new_seg = np.row_stack((last, [x, val]))
                seg.append(new_seg)
                collection.set_segments(seg)

        if self.period_summary_func:
            text_dict = self.add_period_summary(ax, s_all)
            text = self.texts['__period_summary_func__']
            x_period, y_period, text_val = text_dict.pop('x'), text_dict.pop('y'), text_dict.pop('s')
            text.set_position((x_period, y_period))
            text.set_text(text_val)

        if self.images:
            for col in self.df_values.columns:
                xpixel, ypixel = ax.transData.transform((x, y[col]))
                center = xpixel, ypixel
                left, right = xpixel - self.image_radius, xpixel + self.image_radius
                bottom, top = ypixel - self.image_radius, ypixel + self.image_radius
                img, circle = self.images[col]
                img.set_extent([left, right, bottom, top])
                circle.set_center(center)
                img.set_clip_path(circle)
                vis = visible[col]
                img.set_visible(vis)

    def init_func(self):
        ax = self.fig.axes[0]
        s = self.df_values.iloc[0] # current Series
        s_all = self.all_values.iloc[0]
        if len(self.df_others) > 0:
            s_others = self.df_others.iloc[0]
        x, y = s.name, s.to_dict()
        if self.is_x_date:
            x = mdates.date2num(x)

        if self.images:
            xmin, xmax = ax.get_xlim()
            x_extra = (xmax - xmin) * .025
        else:
            x_extra = 0

        visible = self.get_visible(0)
        
        for col in self.df_values.columns:
            val = y[col]
            vis = visible[col]
            color = self.colors[col]
            text = ax.text(x + x_extra, val, col, visible=vis, **self.line_label_font)
            lw = self.line_kwargs['lw']
            ls = self.line_kwargs['ls']
            alpha = self.line_kwargs.get('alpha', 1)
            color[-1] = alpha
            if self.line_width_data is not None:
                lw = self.line_width_data.iloc[0][col]
            lc = LineCollection([[(x, val)]], colors=[color], visible=vis, linewidths=[lw], linestyles=[ls])
            collection = ax.add_collection(lc)
            self.texts[col] = text
            self.collections[col] = collection

        if self.others_line_func is True:
            y_other = self.df_others.iloc[0].to_dict()
            for col, val in y_other.items():
                collection = ax.add_collection(LineCollection([[(x, val)]], colors=[OTHERS_COLOR]))
                self.other_collections[col] = collection

        if self.agg_line is not None:
            color = self.agg_line_kwargs['color']
            lw = self.agg_line_kwargs.get('lw')
            ls = self.agg_line_kwargs.get('ls')
            alpha = self.agg_line_kwargs.get('alpha', AGG_COLOR[-1])
            color = tuple(color[:3]) + (alpha,)
            label = self.agg_line_label
            val = self.agg_line.iloc[0]
            lc = LineCollection([[(x, val)]], colors=[color], linewidths=[lw], linestyles=[ls])
            collection = ax.add_collection(lc)
            text = ax.text(x + x_extra, val, label, **self.line_label_font)

            label = '___agg_line___'
            self.collections[label] = collection
            self.texts[label] = text
            self.colors[label] = color

        if self.others_agg_line is not None:
            color = self.others_line_kwargs['color']
            lw = self.others_line_kwargs.get('lw')
            ls = self.others_line_kwargs.get('ls')
            alpha = self.others_line_kwargs.get('alpha', OTHERS_COLOR[-1])
            color = tuple(color[:3]) + (alpha,)
            label = self.others_line_label
            val = self.others_agg_line.iloc[0]
            lc = LineCollection([[(x, val)]], colors=[color], linewidths=[lw], linestyles=[ls])
            collection = ax.add_collection(lc)
            text = ax.text(x + x_extra, val, label, ha='left', va='center', size='smaller')

            label = '___others_line___'
            self.collections[label] = collection
            self.texts[label] = text
            self.colors[label] = color

        if self.period_summary_func:
            text_dict = self.add_period_summary(ax, s_all)
            text = ax.text(transform=ax.transAxes, **text_dict)
            self.texts['__period_summary_func__'] = text

        if self.images:
            for col in self.df_values.columns:
                xpixel, ypixel = ax.transData.transform((x, y[col]))
                center = xpixel, ypixel
                circle = mpatches.Circle(center, transform=None, radius=self.image_radius, fill=None, lw=0)
                ax.add_patch(circle)
                left, right = xpixel - self.image_radius, xpixel + self.image_radius
                bottom, top = ypixel - self.image_radius, ypixel + self.image_radius
                img_array = self.images[col]
                img = ax.imshow(img_array, extent=[left, right, bottom, top], aspect='auto', transform=None, zorder=4)
                img.set_clip_path(circle)
                vis = visible[col]
                img.set_visible(vis)
                self.images[col] = img, circle

    def make_animation(self):
        interval = self.period_length / self.steps_per_period
        pause = int(self.end_period_pause // interval)

        def frame_generator(n):
            frames = []
            for i in range(1, n):
                frames.append(i)
                if pause and i % self.steps_per_period == 0 and i != 0 and i != n - 1:
                    for _ in range(pause):
                        frames.append(None)
            return frames
        
        frames = frame_generator(len(self.df_values))
        anim = FuncAnimation(self.fig, self.anim_func, frames, self.init_func, interval=interval)

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
                
                ret_val = anim.save(self.filename, fps=self.fps, writer=self.writer, 
                                    savefig_kwargs=savefig_kwargs) 
        except Exception as e:
            message = str(e)
            raise Exception(message)
        finally:
            plt.rcParams = self.orig_rcParams

        return ret_val


def line_chart_race(df, filename=None, n_lines=None, steps_per_period=10, 
                    period_length=500, end_period_pause=0, period_summary_func=None, 
                    line_width_data=None, agg_line_func=None, agg_line_kwargs=None, 
                    others_line_func=None, others_line_kwargs=None, fade=1, min_fade=.3, 
                    images=None, colors=None, title=None, line_label_font=None, 
                    tick_label_font=None, tick_template='{x:,.0f}', shared_fontdict=None, 
                    scale='linear', fig=None, writer=None, line_kwargs=None, 
                    fig_kwargs=None):
    '''
    Create an animated line chart race using matplotlib. Data must be in 
    'wide' format where each row represents a single time period and each 
    column represents a distinct category. Optionally, the index can label 
    the time period.

    If no `filename` is given, an HTML string is returned, otherwise the 
    animation is saved to disk.

    You must have ffmpeg installed on your machine to save videos to disk
    and ImageMagick to save animated gifs. Read more here:
    https://www.dexplo.org/bar_chart_race/installation/

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'wide' DataFrame where each row represents a single period 
        of time. Each column contains the values of the lines for that 
        category. Optionally, use the index to label each time period.
        The index can be of any type.

    filename : `None` or str, default None
        If `None` return animation as an HTML5 string. If a string, save 
        animation to that filename location. Use .mp4, .gif, .html, .mpeg, 
        .mov or any other extensions supported by ffmpeg or ImageMagick.

    n_lines : int, default None
        The maximum number of lines to display on the graph. 
        When there are more columns than n_lines, the columns 
        chosen will be those with the highest values in the last 
        period. 
        
        See the others_line_func parameter for more options on plotting 
        the other columns outside of the top n_lines. Use all lines by default.

    steps_per_period : int, default 10
        The number of steps to go from one time period to the next. 
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

    period_summary_func : function, default None
        Custom text added to the axes each period.
        Create a user-defined function that accepts one pandas Series of the 
        current time period's values. It must return a dictionary containing 
        the keys "x", "y", and "s" which will be passed to the matplotlib 
        `text` method.
        Example:
        def func(values):
            total = values.sum()
            s = f'Worldwide deaths: {total}'
            return {'x': .05, 'y': .85, 's': s, 'size': 10}

    line_width_data : DataFrame, default None
        Control the width of the line at each time period.
        Provide a separate DataFrame with the same index, columns, and 
        dimensions as the original. Line width will be scaled so that
        it is between 1 and 6 points.

    agg_line_func : function or str, default None
        Create an additional line to summarize all of the data. Pass it either 
        a string that the DataFrame `agg` method understands or a user-defined 
        function. Use agg_line_kwargs to style the line and provide it a label.

        If providing function, it will be passed all values of the current 
        period as a Series. Return a single value that summarizes the current 
        period.

        DataFrame agg strings - 'mean', 'median', 'max', 'min', etc..

    agg_line_kwargs : dict, default None
        A dictionary of matplotlib line properties used with agg_line_func. 
        Use the key `s` to control the label of the line. Keys `x` and `y` 
        will be ignored as the position is already determined.
        Example: 
        {
            's': 'Median',
            'color': '.2',
            'lw': 3,
            'ls': '--',
        }

    others_line_func : bool, str, or function, default None
        This parameter may be used when there are more columns than n_lines.
        By default (None), these other lines will not be plotted. Use True to 
        plot the lines in a soft gray color without labels or images.

        Aggregate all of the other column values by passing in a string that 
        the DataFrame agg method understands or a user-defined function, which 
        will be passed a pandas Series of just these other values for this 
        time period.
        Example:
        def my_others_line_func(s):
            return s.median()

    others_line_kwargs : dict, default None
        A dictionary of matplotlib line properties used with others_line_func. 
        Use the key `s` to control the label of the line. Keys `x` and `y` 
        will be ignored as the position is already determined.
        Example: 
        {
            's': 'Rest of World',
            'color': '.2',
            'lw': 3,
            'ls': '--',
        }

    fade : float, default 1
        Use to slowly fade historical values of the line, i.e. decrease the 
        opacity (alpha). This number multiplies the current alpha of the line 
        each time period.
        
        Choose a number between 0 and 1. When 1, no fading occurs. 
        This number will likely need to be close to 1, as alpha decreases fast.
        
    min_fade : float, default .3
        Minimum value of alpha for each line when setting fade < 1. Choose a 
        number between 0 and 1. Use 0 to have the line eventually become 
        completely transparent.
    
    images : str, list, or dict, default None
        Images to use for the end of the line. Use a string to use one of the
        built-in image databases by name. Currently, there are two image bases
        'country', and 'nfl'. If your columns are countries are NFL teams, the 
        images will automatically be found.
        
        Otherwise, provide a list of filenames or URLs where the images are 
        located in the same order as the columns of the DataFrame or a 
        dictionary mapping column names to filenames/URLs.

    colors : str, matplotlib colormap instance, or list of colors, default 'dark12'
        Colors to be used for the lines. All matplotlib and plotly 
        colormaps are available by string name. Colors will repeat 
        if there are more lines than colors.

        'dark12' is a discrete colormap. If there are more than 12 columns, 
        then the default colormap will be 'dark24'

        Append "_r" to the colormap name to use the reverse of the colormap.
        i.e. "dark12_r"

    title : str or dict, default None
        Title of plot as a string. Use a dictionary to supply several title 
        parameters. You must use the key 'label' for the text.
        Example:
        {
            'label': 'My Line Chart Race Title',
            'size': 18,
            'color': 'red',
            'loc': 'right',
            'pad': 12
        }

    line_label_font : number or dict, default None
        Font size of labels at the end of the lines. When None, defaults to 7. 
        Use a dictionary to supply several font properties.

    tick_label_font : number or dict, default None
        Font size of tick labels. When None, defaults to 7.
        Use a dictionary to supply several font properties.

    tick_template : str or function, default '{x:,.0f}'
        Formats the ticks on the y-axis with numeric values When given a 
        string, it's passed to the ticker.StrMethodFormatter function. 
        Use 'x' as the variable
        Example: '{x:10.2f}'

        WHen given a function, it's passed to ticker.FuncFormatter, which
        implicitly passes it two parameters `x` and `pos` and must return
        a string.

    shared_fontdict : dict, default None
        Dictionary of font properties shared across the tick labels, 
        line labels, and title. The only property not shared is `size`. 
        It will be ignored if you try to set it.
        Possible keys are:
            'family', 'weight', 'color', 'style', 'stretch', 'weight', 'variant'
        Example:
        {
            'family' : 'Helvetica',
            'weight' : 'bold',
            'color' : 'rebeccapurple'
        }

    scale : 'linear' or 'log', default 'linear'
        Type of scaling to use for the y-axis

    fig : matplotlib Figure, default None
        For greater control over the aesthetics, supply your own figure
        with at least one axes.

    writer : str or matplotlib Writer instance
        This argument is passed to the matplotlib FuncAnimation.save method.

        By default, the writer will be 'ffmpeg' unless creating a gif,
        then it will be 'imagemagick', or an html file, then it 
        will be 'html'. 
            
        Find all of the availabe Writers:
        >>> from matplotlib import animation
        >>> animation.writers.list()

    line_kwargs : dict, default None
        Other keyword arguments within a dictionary used to control line
        properties. 
        Sample properties:
            `lw` - Line width, default is 1.5
            'ls' - Line style, '-', '--', '-.', ':'
            `alpha` - opacity of line, 0 to 1

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

    Returns
    -------
    When `filename` is left as `None`, an HTML5 video is returned as a string.
    Otherwise, a file of the animation is saved and `None` is returned.

    Examples
    --------
    Use the `load_data` function to get an example dataset to 
    create an animation.

    df = bcr.load_dataset('covid19')
    bcr.line_chart_race(
        df=df, 
        filename='covid19_line_race.mp4', 
        n_lines=5, 
        steps_per_period=10, 
        period_length=500, 
        end_period_pause=300,
        period_summary_func=lambda v, r: {'x': .85, 'y': .2, 
                                          's': f'Total deaths: {v.sum()}', 
                                          'ha': 'right', 'size': 11}, 
        line_width_data=None,
        agg_line_func='median', 
        agg_line_kwargs=None
        others_line_func=None,
        others_line_kwargs=None,
        fade=.99,
        min_fade=.5,
        colors='dark12', 
        title='COVID-19 Deaths by Country', 
        line_label_font=7, 
        tick_label_font=7, 
        tick_template='{x:,.0f}'
        shared_fontdict={'family' : 'Helvetica', 'weight' : 'bold', 'color' : '.1'}, 
        scale='linear', 
        fig=None, 
        writer=None, 
        line_kwargs={'alpha': .7},
        fig_kwargs={'figsizse': (6, 3.5), 'dpi': 144})        

    Font Help
    ---------
    Font size can also be a string - 'xx-small', 'x-small', 'small',  
        'medium', 'large', 'x-large', 'xx-large', 'smaller', 'larger'
    These sizes are relative to plt.rcParams['font.size'].
    '''

    lcr = _LineChartRace(df, filename, n_lines, steps_per_period, period_length, end_period_pause, 
                         period_summary_func, line_width_data, agg_line_func, agg_line_kwargs, 
                         others_line_func, others_line_kwargs, fade, min_fade, images, colors, 
                         title, line_label_font, tick_label_font, tick_template, shared_fontdict, 
                         scale, fig, writer, line_kwargs, fig_kwargs)
    return lcr.make_animation()
