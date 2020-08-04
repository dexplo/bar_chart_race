import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly

from ._utils import prepare_wide_data


class _BarChartRace:
    
    def __init__(self, df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                 steps_per_period, period_length, end_period_pause, interpolate_period, 
                 period_label, period_template, period_summary_func, perpendicular_bar_func, 
                 colors, title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                 tick_label_font, hovertemplate, slider, scale, bar_kwargs, layout_kwargs, 
                 write_html_kwargs, filter_column_colors):
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
        self.bar_texttemplate = self.get_bar_texttemplate(bar_texttemplate)
        self.bar_label_font = self.get_font(bar_label_font)
        self.tick_label_font = self.get_font(tick_label_font)
        self.hovertemplate = self.get_hovertemplate(hovertemplate)
        self.slider = slider
        self.scale = scale
        self.duration = self.period_length / steps_per_period
        self.write_html_kwargs = write_html_kwargs or {}
        self.filter_column_colors = filter_column_colors
        
        self.validate_params()
        self.bar_kwargs = self.get_bar_kwargs(bar_kwargs)
        self.layout_kwargs = self.get_layout_kwargs(layout_kwargs)
        self.df_values, self.df_ranks = self.prepare_data(df)
        self.col_filt = self.get_col_filt()
        self.bar_colors = self.get_bar_colors(colors)
        self.set_fixed_max_limits()
        self.str_index = self.df_values.index.astype('str')

    def get_extension(self):
        if self.filename:
            return self.filename.split('.')[-1]

    def get_bar_texttemplate(self, bar_texttemplate):
        if bar_texttemplate is None:
            bar_texttemplate = '%{x:,.0f}' if self.orientation == 'h' else '%{y:,.0f}'
        return bar_texttemplate

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

    def get_bar_kwargs(self, bar_kwargs):
        if bar_kwargs is None:
            return {'opacity': .8}
        elif isinstance(bar_kwargs, dict):
            if 'opacity' not in bar_kwargs:
                bar_kwargs['opacity'] = .8
            return bar_kwargs
        raise TypeError('`bar_kwargs` must be None or a dictionary mapping `go.Bar` parameters '
                        'to values.')

    def get_layout_kwargs(self, layout_kwargs):
        if layout_kwargs is None:
            return {'showlegend': False}
        elif isinstance(layout_kwargs, dict):
            if {'xaxis', 'yaxis', 'annotations'} & layout_kwargs.keys():
                raise ValueError('`layout_kwargs` cannot contain "xaxis", "yaxis", or '
                                 ' "annotations".')
            if 'showlegend' not in layout_kwargs:
                layout_kwargs['showlegend'] = False
            return layout_kwargs
        elif isinstance(layout_kwargs, plotly.graph_objs._layout.Layout):
            return self.get_layout_kwargs(layout_kwargs.to_plotly_json())
        raise TypeError('`layout_kwargs` must be None, a dictionary mapping '
                        '`go.Layout` parameters to values or an instance of `go.Layout`.')

    def get_period_label(self, period_label):
        if period_label is False:
            return False

        default_period_label = {'xref': 'paper', 'yref': 'paper', 'font': {'size': 20},
                                'xanchor': 'right', 'showarrow': False}
        if self.orientation == 'h':
            default_period_label['x'] = .95
            default_period_label['y'] = .15 if self.sort == 'desc' else .85
        else:
            default_period_label['x'] = .95 if self.sort == 'desc' else .05
            default_period_label['y'] = .85
            default_period_label['xanchor'] = 'left' if self.sort == 'asc' else 'right'

        if period_label is True:
            return default_period_label
        elif isinstance(period_label, dict):
            period_label = {**default_period_label, **period_label}
        else:
            raise TypeError('`period_label` must be a boolean or dictionary')

        return period_label

    def get_title(self, title):
        if title is None:
            return
        if isinstance(title, str):
            return {'text': title, 'y': 1, 'x': .5, 'xref': 'paper', 'yref': 'paper',
                    'pad': {'b': 10},
                    'xanchor': 'center', 'yanchor': 'bottom'}
        elif isinstance(title, (dict, plotly.graph_objects.layout.Title)):
            return title
        raise TypeError('`title` must be a string, dictionary, or '
                        '`plotly.graph_objects.layout.Title` instance')

    def get_font(self, font):
        if font is None:
            font = {'size': 12}
        elif isinstance(font, (int, float)):
            font = {'size': font}
        elif not isinstance(font, dict):
            raise TypeError('`font` must be a number or dictionary of font properties')
        return font

    def get_hovertemplate(self, hovertemplate):
        if hovertemplate is None:
            if self.orientation == 'h':
                return '%{y} - %{x:,.0f}<extra></extra>'
            return '%{x} - %{y:,.0f}<extra></extra>'
        return hovertemplate
            
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
        dfs = prepare_wide_data(df, orientation=self.orientation, sort=self.sort, 
                                n_bars=self.n_bars, interpolate_period=self.interpolate_period, 
                                steps_per_period=self.steps_per_period, compute_ranks=compute_ranks)
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
        elif isinstance(colors, list):
            bar_colors = colors
        elif isinstance(colors, tuple):
            bar_colors = list(colors)
        elif hasattr(colors, 'tolist'):
            bar_colors = colors.tolist()
        else:
            raise TypeError('`colors` must be a string name of a colormap or '
                            'sequence of colors.')

        # bar_colors is now a list
        n = len(bar_colors)
        orig_bar_colors = bar_colors
        if self.df_values.shape[1] > n:
            bar_colors = bar_colors * (self.df_values.shape[1] // n + 1)
        bar_colors = np.array(bar_colors[:self.df_values.shape[1]])

        # plotly uses 0, 255 rgb colors, matplotlib is 0 to 1
        if bar_colors.dtype.kind == 'f' and bar_colors.shape[1] == 3 and  (bar_colors <= 1).all():
            bar_colors = pd.DataFrame(bar_colors).astype('str')
            bar_colors = bar_colors.apply(lambda x: ','.join(x), axis = 1)
            bar_colors = ('rgb(' + bar_colors + ')').values
        
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

    def set_fixed_max_limits(self):
        label_limit = (.2, self.n_bars + .8)
        value_limit = None
        min_val = 1 if self.scale == 'log' else 0
        if self.fixed_max:
            value_limit = [min_val, self.df_values.max().max() * 1.1]

        if self.orientation == 'h':
            self.xlimit = value_limit
            self.ylimit = label_limit
        else:
            self.xlimit = label_limit
            self.ylimit = value_limit
        
    def set_value_limit(self, bar_vals):
        min_val = 1 if self.scale == 'log' else 0
        if not self.fixed_max:
            value_limit = [min_val, bar_vals.max() * 1.1]

            if self.orientation == 'h':
                self.xlimit = value_limit
            else:
                self.ylimit = value_limit
  
    def get_frames(self):
        frames = []
        slider_steps = []
        for i in range(len(self.df_values)):
            bar_locs = self.df_ranks.iloc[i].values
            top_filt = (bar_locs >= 0) & (bar_locs < self.n_bars + 1)
            bar_vals = self.df_values.iloc[i].values
            bar_vals[bar_locs == 0] = 0
            bar_vals[bar_locs == self.n_bars + 1] = 0
            # self.set_value_limit(bar_vals) # plotly bug? not updating range
            
            cols = self.df_values.columns.values.copy()
            cols[bar_locs == 0] = ' '
            colors = self.bar_colors
            bar_locs = bar_locs + np.random.rand(len(bar_locs)) / 10_000 # done to prevent stacking of bars
            x, y = (bar_vals, bar_locs) if self.orientation == 'h' else (bar_locs, bar_vals)

            label_axis = dict(tickmode='array', tickvals=bar_locs, ticktext=cols, 
                              tickfont=self.tick_label_font)

            label_axis['range'] = self.ylimit if self.orientation == 'h' else self.xlimit
            if self.orientation == 'v':
                label_axis['tickangle'] = -90

            value_axis = dict(showgrid=True, type=self.scale)#, tickformat=',.0f')
            value_axis['range'] = self.xlimit if self.orientation == 'h' else self.ylimit

            bar = go.Bar(x=x, y=y, width=self.bar_size, textposition=self.bar_textposition,
                         texttemplate=self.bar_texttemplate, orientation=self.orientation, 
                         marker_color=colors, insidetextfont=self.bar_label_font, 
                         cliponaxis=False, outsidetextfont=self.bar_label_font, 
                         hovertemplate=self.hovertemplate, **self.bar_kwargs)

            data = [bar]
            xaxis, yaxis = (value_axis, label_axis) if self.orientation == 'h' \
                             else (label_axis, value_axis)

            annotations = self.get_annotations(i)
            if self.slider and i % self.steps_per_period == 0:
                slider_steps.append(
                            {"args": [[i],
                                {"frame": {"duration": self.duration, "redraw": False},
                                 "mode": "immediate",
                                 "fromcurrent": True,
                                 "transition": {"duration": self.duration}
                                }],
                            "label": self.get_period_label_text(i), 
                            "method": "animate"})
            layout = go.Layout(xaxis=xaxis, yaxis=yaxis, annotations=annotations, 
                                margin={'l': 150}, **self.layout_kwargs)
            if self.perpendicular_bar_func:
                pbar = self.get_perpendicular_bar(bar_vals, i, layout)
                layout.update(shapes=[pbar], overwrite=True)
            frames.append(go.Frame(data=data, layout=layout, name=i))

        return frames, slider_steps

    def get_period_label_text(self, i):
        if self.period_template:
            idx_val = self.df_values.index[i]
            if self.df_values.index.dtype.kind == 'M':
                s = idx_val.strftime(self.period_template)
            else:
                s = self.period_template.format(x=idx_val)
        else:
            s = self.str_index[i]
        return s
    
    def get_annotations(self, i):
        annotations = []
        if self.period_label:
            self.period_label['text'] = self.get_period_label_text(i)
            annotations.append(self.period_label)

        if self.period_summary_func:
            values = self.df_values.iloc[i]
            ranks = self.df_ranks.iloc[i]
            text_dict = self.period_summary_func(values, ranks)
            if 'x' not in text_dict or 'y' not in text_dict or 'text' not in text_dict:
                name = self.period_summary_func.__name__
                raise ValueError(f'The dictionary returned from `{name}` must contain '
                                  '"x", "y", and "s"')
            text, x, y = text_dict['text'], text_dict['x'], text_dict['y']
            annotations.append(dict(text=text, x=x, y=y, font=dict(size=14), 
                                    xref="paper", yref="paper", showarrow=False))

        return annotations

    def get_perpendicular_bar(self, bar_vals, i, layout):
        if isinstance(self.perpendicular_bar_func, str):
            val = pd.Series(bar_vals).agg(self.perpendicular_bar_func)
        else:
            values = self.df_values.iloc[i]
            ranks = self.df_ranks.iloc[i]
            val = self.perpendicular_bar_func(values, ranks)

        xref, yref = ("x", "paper") if self.orientation == 'h' else ("paper", "y")
        value_limit = self.xlimit if self.orientation == 'h' else self.ylimit
        if self.fixed_max:
            delta = (value_limit[1] - value_limit[0]) * .02
        else:
            delta = (1.05 * bar_vals.max() - bar_vals.min()) * .02

        x0, x1 = (val - delta, val + delta) if self.orientation == 'h' else (0, 1)
        y0, y1 = (val - delta, val + delta) if self.orientation == 'v' else (0, 1)

        return dict(type="rect", xref=xref, yref=yref, x0=x0, y0=y0, x1=x1, y1=y1,
                    fillcolor="#444444",layer="below", opacity=.5, line_width=0)

    def make_animation(self):
        frames, slider_steps = self.get_frames()
        data = frames[0].data
        layout = frames[0].layout
        layout.title = self.title
        layout.updatemenus = [dict(
            type="buttons",
            direction = "left",
            x=1, 
            y=1.02,
            xanchor='right',
            yanchor='bottom',
            buttons=[dict(label="Play",
                          method="animate",
                          # redraw must be true for bar plots
                          args=[None, {"frame": {"duration": self.duration, "redraw": True},
                                        "fromcurrent": True
                                    }]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}]),
                     ]
                     )]

        sliders_dict = {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            # "font": {"size": 20},
                            # "prefix": '', # allow user to set
                            "visible": False, # just repeats period label
                            # "xanchor": "right"
                        },
                        "transition": {"duration": self.duration, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.88,
                        "x": 0.05,
                        "y": 0,
                        "steps": slider_steps
                    }
        if self.slider:
            layout.sliders = [sliders_dict]

        fig = go.Figure(data=data, layout=layout, frames=frames[1:])
        if self.filename:
            fig.write_html(self.filename, **self.write_html_kwargs)
        else:
            return fig


def bar_chart_race_plotly(df, filename=None, orientation='h', sort='desc', n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=False, 
                          period_label=True, period_template=None, period_summary_func=None, 
                          perpendicular_bar_func=None, colors=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, scale='linear', 
                          bar_kwargs=None, layout_kwargs=None, write_html_kwargs=None, 
                          filter_column_colors=False):
    '''
    Create an animated bar chart race using Plotly. Data must be in 
    'wide' format where each row represents a single time period and each 
    column represents a distinct category. Optionally, the index can label 
    the time period. Bar length and location change linearly from one time 
    period to the next.

    Note - The duration of each frame is calculated as 
    `period_length` / `steps_per_period`, but is unlikely to actually 
    be this number, especially when duration is low (< 50ms). You may have to
    experiment with different combinations of `period_length` and
    `steps_per_period` to get the animation at the desired speed.

    If no `filename` is given, a plotly figure is returned that is embedded
    into the notebook.

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'wide' DataFrame where each row represents a single period 
        of time. Each column contains the values of the bars for that 
        category. Optionally, use the index to label each time period.
        The index can be of any type.

    filename : `None` or str, default None
        If `None` return plotly animation, otherwise save
        to disk. Can only save as HTML at this time.

    orientation : 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort : 'desc' or 'asc', default 'desc'
        Choose how to sort the bars. Use 'desc' to put largest bars on top 
        and 'asc' to place largest bars on bottom.

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
        When `False`, the axis for the values will have its maximum (x/y)
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

    period_length : int, default 500
        Number of milliseconds to animate each period (row). 
        Default is 500ms (half of a second)

    end_period_pause : int, default 0
        Number of milliseconds to pause the animation at the end of
        each period.

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
        on the figure labeling each period. No label when 'False'.

        Use a dictionary to supply the exact position of the period
        along with any valid parameters of a plotly annotation.

        Example:
        {
            'x': .99,
            'y': .8,
            'font' : {'family': 'Helvetica', 'size': 20, 'color': 'orange'},
            'xanchor': 'right',
        }
        
        Reference - https://plotly.com/python/reference/#layout-annotations

        The default location depends on `orientation` and `sort`
        * h, desc -> x=.95, y=.15
        * h, asc -> x=.95, y=.85
        * v, desc -> x=.95, y=.85
        * v, asc -> x=.05, y=.85

    period_template : str, default `None`
        Either a string with date directives or 
        a new-style (Python 3.6+) formatted string

        For a string with a date directive, find the complete list here
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        
        Example of string with date directives
            '%B %d, %Y'
        Will change 2020/03/29 to March 29, 2020
        
        For new-style formatted string. Use curly braces and the variable `x`, 
        which will be passed the current period's index value.
        Example:
            'Period {x:10.2f}'

        Date directives will only be used for datetime indexes.

    period_summary_func : function, default None
        Custom text added to the axes each period.
        Create a user-defined function that accepts two pandas Series of the 
        current time period's values and ranks. It must return a dictionary 
        containing at a minimum the keys "x", "y", and "text" which will be 
        passed used for a plotly annotation.

        Example:
        def func(values, ranks):
            total = values.sum()
            text = f'Worldwide deaths: {total}'
            return {'x': .85, 'y': .2, 'text': text, 'size': 11}

    perpendicular_bar_func : function or str, default None
        Creates a single bar perpendicular to the main bars that spans the 
        length of the axis. 
        
        Use either a string that the DataFrame `agg` method understands or a 
        user-defined function.
            
        DataFrame strings - 'mean', 'median', 'max', 'min', etc..

        The function is passed two pandas Series of the current time period's
        data and ranks. It must return a single value.

        def func(values, ranks):
            return values.quantile(.75)

    colors : str or sequence colors, default 'dark12'
        Colors to be used for the bars. All matplotlib and plotly colormaps are 
        available by string name. Colors will repeat if there are more bars than colors.

        'dark12' is the default colormap. If there are more than 10 columns, 
        then the default colormap will be 'dark24'

        Append "_r" to the colormap name to use the reverse of the colormap.
        i.e. "dark12_r"

    title : str, dict, or plotly.graph_objects.layout.Title , default None
        Title of animation. Use a string for simple titles or a
        dictionary to specify several properties
        {'text': 'My Bar Chart Race', 
         'x':0.5, 
         'y':.9,
         'xanchor': 'center', 
         'yanchor': 'bottom'}

        Other properties include: font, pad, xref, yref

    bar_size : float, default .95
        Height/width of bars for horizontal/vertical bar charts. 
        Use a number between 0 and 1
        Represents the fraction of space that each bar takes up. 
        When equal to 1, no gap remains between the bars.

    bar_textposition : str or sequence, default `None`
        Position on bar to place its label.
        Use one of the strings - 'inside', 'outside', 'auto', 'none'
        or a sequence of the above

    bar_texttemplate : str, default '%{x:,.0f}' or '%{y:,.0f}'
        Template string used for rendering the text inside/outside
        the bars. Variables are inserted using %{variable},
        for example "y: %{y}". Numbers are formatted using
        d3-format's syntax %{variable:d3-format}, for example
        "Price: %{y:$.2f}".

    bar_label_font : number or dict, None
        Font size of numeric bar labels. When None, font size is 12. 
        Use a dictionary to supply several font properties.
        Example:
        {
            'size': 12,
            'family': 'Courier New, monospace',
            'color': '#7f7f7f'
        }

    tick_label_font : number or dict, None
        Font size of tick labels.When None, font size is 12. 
        Use a dictionary to supply several font properties.

    hovertemplate : str, default None
        Template string used for rendering the information that appear 
        on hover box. By default, it is '%{y} - %{x:,.0f}<extra></extra>'

        Reference: https://plotly.com/python/hover-text-and-formatting

    slider : bool, default True
        Whether or not to place a slider below the animation

    scale : 'linear' or 'log', default 'linear'
        Type of scaling to use for the axis containing the values

    bar_kwargs : dict, default `None` (opacity=.8)
        Other keyword arguments (within a dictionary) forwarded to the 
        plotly `go.Bar` function. If no value for 'opacity' is given,
        then it is set to .8 by default.

    layout_kwargs : dict or go.Layout instance, default None
        Other keyword arguments (within a dictionary) are forwarded to 
        the plotly `go.Layout` function. Use this to control the size of
        the figure.
        Example:
        {
            'width': 600,
            'height': 400,
            'showlegend': True
        }

    write_html_kwargs : dict, default None
        Arguments passed to the write_html plotly go.Figure method.
        Example:
        {
            'auto_play': False,
            'include_plotlyjs': 'cdn',
            'full_html': False=
        }
        Reference: https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html
                   
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
    When `filename` is left as `None`, a plotly figure is returned and
    embedded into the notebook. Otherwise, a file of the HTML is 
    saved and `None` is returned.

    References
    -----
    Plotly Figure - https://plotly.com/python/reference
    Plotly API - https://plotly.com/python-api-reference
    d3 formatting - https://github.com/d3/d3-3.x-api-reference/blob/master/Formatting.md
    
    Examples
    --------
    Use the `load_data` function to get an example dataset to 
    create an animation.

    df = bcr.load_dataset('covid19')
    bcr.bar_chart_race_plotly(
        df=df, 
        filename='covid19_horiz_desc.html', 
        orientation='h', 
        sort='desc', 
        n_bars=8, 
        fixed_order=False, 
        fixed_max=True, 
        steps_per_period=10, 
        period_length=500, 
        interpolate_period=False, 
        period_label={'x': .99, 'y': .8, 'font': {'size': 25, 'color': 'blue'}}, 
        period_template='%B %d, %Y', 
        period_summary_func=lambda v, r: {'x': .85, 'y': .2, 
                                          's': f'Total deaths: {v.sum()}', 
                                          'size': 11}, 
        perpendicular_bar_func='median', 
        colors='dark12', 
        title='COVID-19 Deaths by Country', 
        bar_size=.95,
        bar_textposition='outside', 
        bar_texttemplate='%{x}',
        bar_label_font=12, 
        tick_label_font=12, 
        hovertemplate=None,
        scale='linear', 
        bar_kwargs={'opacity': .7},
        write_html_kwargs=None,
        filter_column_colors=False)        
    '''
    bcr = _BarChartRace(df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                        steps_per_period, period_length, end_period_pause, interpolate_period, 
                        period_label, period_template, period_summary_func, perpendicular_bar_func, 
                        colors, title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                        tick_label_font, hovertemplate, slider, scale, bar_kwargs, layout_kwargs, 
                        write_html_kwargs, filter_column_colors)
    return bcr.make_animation()
