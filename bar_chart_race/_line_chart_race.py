import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker, colors as mcolors, dates as mdates
from matplotlib.collections import LineCollection

from ._utils import prepare_wide_data

class _LineChartRace:
    
    def __init__(self, df, filename, orientation, sort, n_lines, fixed_max,
                 steps_per_period, period_length, interpolate_period, period_label, 
                 period_fmt, period_summary_func, agg_line_func, figsize, cmap, title, 
                 tick_label_size, shared_fontdict, scale, writer, fig, dpi, 
                 line_kwargs, filter_column_colors):
        self.filename = filename
        self.extension = self.get_extension()
        self.orientation = orientation
        self.sort = sort
        self.n_lines = n_lines or df.shape[1]
        self.fixed_max = fixed_max
        self.steps_per_period = steps_per_period
        self.interpolate_period = interpolate_period
        self.period_label = self.get_period_label(period_label)
        self.period_fmt = period_fmt
        self.period_summary_func = period_summary_func
        self.agg_line_func = agg_line_func
        self.period_length = period_length
        self.figsize = figsize
        self.title = self.get_title(title)
        self.tick_label_size = tick_label_size
        self.orig_rcParams = self.set_shared_fontdict(shared_fontdict)
        self.scale = scale
        self.writer = self.get_writer(writer)
        self.fps = 1000 / self.period_length * steps_per_period
        self.filter_column_colors = filter_column_colors
        
        self.validate_params()
        self.line_kwargs = self.get_line_kwargs(line_kwargs)
        self.html = self.filename is None
        self.df_values = self.prepare_data(df)
        self.fig, self.ax = self.get_fig(fig, dpi)
        self.colors = self.get_colors(cmap)
        self.str_index = self.df_values.index.astype('str')
        

    def get_extension(self):
        if self.filename:
            return self.filename.split('.')[-1]

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

    def get_line_kwargs(self, line_kwargs):
        line_kwargs = line_kwargs or {}
        if 'alpha' not in line_kwargs:
            line_kwargs['alpha'] = .8
        if 'ec' not in line_kwargs:
            line_kwargs['ec'] = 'white'
        return line_kwargs

    def get_period_label(self, period_label):
        if not period_label:
            return False
        elif period_label is True:
            period_label = {'size': 12}
            if self.orientation == 'h':
                period_label['x'] = .95
                period_label['y'] = .15 if self.sort == 'desc' else .85
                period_label['ha'] = 'right'
                period_label['va'] = 'center'
            else:
                period_label['x'] = .95 if self.sort == 'desc' else .05
                period_label['y'] = .85
                period_label['ha'] = 'right' if self.sort == 'desc' else 'left'
                period_label['va'] = 'center'
        else:
            if 'x' not in period_label or 'y' not in period_label:
                raise ValueError('`period_label` dictionary must have keys for "x" and "y"')
        return period_label

    def get_title(self, title):
        if isinstance(title, str):
            return {'label': title}
        elif isinstance(title, dict):
            if 'label' not in title:
                raise ValueError('You must use the key "label" in the `title` dictionary '
                                 'to supply the name of the title')
        elif title is None:
            title = {'label': None}
        else:
            raise TypeError('`title` must be either a string or dictionary')

        return title

    def set_shared_fontdict(self, shared_fontdict):
        orig_rcParams = plt.rcParams.copy()
        if shared_fontdict is None:
            return orig_rcParams
        for k, v in shared_fontdict.items():
            if k not in ['fontsize', 'size']:
                if k in ['cursive', 'family', 'fantasy', 'monospace', 'sans-serif', 'serif']:
                        if isinstance(v, str):
                            v = [v]
                if k == 'color':
                    plt.rcParams['text.color'] = v
                    plt.rcParams['xtick.color'] = v
                    plt.rcParams['ytick.color'] = v
                    continue
                try:
                    plt.rcParams[f'font.{k}'] = v
                except KeyError:
                    raise KeyError(f"{k} is not a valid key in `sharedfont_dict`"
                                    "It must be one of "
                                    "'cursive', 'family', 'fantasy', 'monospace',"
                                    "sans-serif', 'serif',"
                                    "'stretch', 'style', 'variant', 'weight'")
        return orig_rcParams

    def get_writer(self, writer):
        if writer is None:
            if self.extension == 'gif':
                writer = 'imagemagick'
            elif self.extension == 'html':
                writer = 'html'
            else:
                writer = plt.rcParams['animation.writer']
        return writer

    def prepare_data(self, df):
        compute_ranks = False
        return prepare_wide_data(df, self.orientation, self.sort, self.n_lines,
                                True, self.steps_per_period,
                                compute_ranks)
        
    def get_colors(self, cmap):
        if cmap is None:
            cmap = 'dark12'
            if self.df_values.shape[1] > 10:
                cmap = 'dark24'
            
        if isinstance(cmap, str):
            from ._colormaps import colormaps
            try:
                colors = colormaps[cmap.lower()]
            except KeyError:
                raise KeyError(f'Colormap {cmap} does not exist. Here are the '
                               f'possible colormaps: {colormaps.keys()}')
        elif isinstance(cmap, mcolors.Colormap):
            colors = cmap(range(cmap.N)).tolist()
        elif isinstance(cmap, list):
            colors = cmap
        elif isinstance(cmap, tuple):
            colors = list(cmap)
        elif hasattr(cmap, 'tolist'):
            colors = cmap.tolist()
        else:
            raise TypeError('`cmap` must be a string name of a colormap, a matplotlib colormap '
                            'instance, list, or tuple of colors')

        # colors is a list
        n = len(colors)
        if self.df_values.shape[1] > n:
            colors = colors * (self.df_values.shape[1] // n + 1)

        colors = mcolors.to_rgba_array(colors)
        colors = colors[:self.df_values.shape[1]]
        return colors

    def get_fig(self, fig, dpi):
        if fig is not None and not isinstance(fig, plt.Figure):
            raise TypeError('`fig` must be a matplotlib Figure instance')
        if fig is not None:
            if not fig.axes:
                raise ValueError('The figure passed to `fig` must have an axes')
            ax = fig.axes[0]
            self.figsize = fig.get_size_inches()
        else:
            fig, ax = self.create_figure(dpi)
        return fig, ax

    def create_figure(self, dpi):
        fig = plt.Figure(figsize=self.figsize, dpi=dpi)
        ax = fig.subplots()
        ax.set_title(**self.title)
        return fig, ax
       
    def init_func(self):
        self.ax.set_xlim(self.df_values.index[0], self.df_values.index[-1])
        self.ax.set_ylim(self.df_values.min().min(), self.df_values.max().max())
        xmin, xmax = self.ax.get_xlim()
        delta = (xmax - xmin) * .05
        self.ax.set_xlim(xmin - delta, xmax + delta)

        ymin, ymax = self.ax.get_ylim()
        delta = (ymax - ymin) * .05
        self.ax.set_ylim(ymin - delta, ymax + delta)

        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.tick_params(length=0)
        self.ax.set_facecolor('.9')
        self.ax.grid(True, axis='y')
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d'))
        
        df_cur = self.df_values.iloc[0]
        x, y = df_cur.name, df_cur.values
        x = mdates.date2num(x)
        for col, val, color in zip(self.df_values.columns, y, self.colors):
            self.ax.text(x, val, col, ha='center', va='bottom')
            self.ax.add_collection(LineCollection([[(x, val)]], colors=[color]))

    def anim_func(self, i):
        df_cur = self.df_values.iloc[i]
        x, y = df_cur.name, df_cur.values
        x = mdates.date2num(x)
        for collection, text, val, color in zip(self.ax.collections, self.ax.texts, y, self.colors):
            seg = collection.get_segments()
            last = seg[-1][-1]
            new_seg = np.row_stack((last, [x, val]))
            seg.append(new_seg)
            collection.set_segments(seg)
            color_arr = collection.get_colors()
            color_arr = np.append(color_arr, [color], axis=0)
            color_arr[:, -1] *= .97
            collection.set_color(color_arr)
            text.set_position((x, val))
        
    def make_animation(self):        
        interval = self.period_length / self.steps_per_period
        anim = FuncAnimation(self.fig, self.anim_func, range(1, len(self.df_values)), 
                             self.init_func, interval=interval)

        try:
            if self.html:
                ret_val = anim.to_html5_video()
                try:
                    from IPython.display import HTML
                    ret_val = HTML(ret_val)
                except ImportError:
                    pass
            else:
                ret_val = anim.save(self.filename, fps=self.fps, writer=self.writer)
        except Exception as e:
            message = str(e)
            raise Exception(message)
        finally:
            plt.rcParams = self.orig_rcParams

        return ret_val

def line_chart_race(df, filename=None, orientation='h', sort='desc', n_lines=None, 
                    fixed_max=True, steps_per_period=10, period_length=500, 
                    interpolate_period=False, period_label=True, period_fmt=None, 
                    period_summary_func=None, agg_line_func=None, figsize=(6, 3.5),
                    cmap=None, title=None, tick_label_size=7, shared_fontdict=None, 
                    scale='linear', writer=None, fig=None, dpi=144, line_kwargs=None, 
                    filter_column_colors=False):
    lcr = _LineChartRace(df, filename, orientation, sort, n_lines, fixed_max,
                         steps_per_period, period_length, interpolate_period, period_label, 
                         period_fmt, period_summary_func, agg_line_func, figsize, cmap, title, 
                         tick_label_size, shared_fontdict, scale, writer, fig, dpi, line_kwargs, 
                         filter_column_colors)
    return lcr.make_animation()
    