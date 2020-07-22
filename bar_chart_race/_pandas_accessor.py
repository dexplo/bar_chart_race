import re

import pandas as pd

from ._bar_chart_race import bar_chart_race as bcr
from ._bar_chart_race_plotly import bar_chart_race_plotly as bcrp
from ._utils import prepare_wide_data as pwd, prepare_long_data as pld


@pd.api.extensions.register_dataframe_accessor("bcr")
class _BCR:

    def __init__(self, df):
        self._df = df

    def bar_chart_race(self, filename=None, orientation='h', sort='desc', n_bars=None, 
                       fixed_order=False, fixed_max=False, steps_per_period=10, 
                       period_length=500, end_period_pause=0, interpolate_period=False, 
                       bar_label_position='outside', bar_label_fmt='{x:,.0f}',
                       bar_size=.95, period_label=True, period_fmt=None, 
                       period_summary_func=None, perpendicular_bar_func=None, 
                       figsize=(6, 3.5), cmap=None, title=None, bar_label_size=7, 
                       tick_label_size=7, shared_fontdict=None, scale='linear', writer=None, 
                       fig=None, dpi=144, bar_kwargs=None, filter_column_colors=False):

        return bcr(self._df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                  steps_per_period, period_length, end_period_pause, interpolate_period, 
                   bar_label_position, bar_label_fmt, bar_size, period_label, period_fmt, 
                   period_summary_func, perpendicular_bar_func, figsize, cmap, title, 
                   bar_label_size, tick_label_size, shared_fontdict, scale, writer, fig, 
                   dpi, bar_kwargs, filter_column_colors)

    def bar_chart_race_plotly(self, filename=None, orientation='h', sort='desc', n_bars=None, 
                              fixed_order=False, fixed_max=False, steps_per_period=10, 
                              period_length=500, interpolate_period=False, period_label=True, 
                              period_fmt=None, period_summary_func=None, perpendicular_bar_func=None,
                              colors=None, title=None, bar_size=.95, textposition='outside', 
                              texttemplate=None, bar_label_font=12, tick_label_font=12, 
                              hovertemplate=None, slider=True, scale='linear', bar_kwargs=None, 
                              layout_kwargs=None, write_html_kwargs=None, filter_column_colors=False):

        return bcrp(self._df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                    steps_per_period, period_length, interpolate_period, period_label, 
                    period_fmt, period_summary_func, perpendicular_bar_func, colors, title, 
                    bar_size, textposition, texttemplate, bar_label_font, tick_label_font, 
                    hovertemplate, slider, scale, bar_kwargs, layout_kwargs, write_html_kwargs,
                    filter_column_colors)

    def prepare_wide_data(self, orientation='h', sort='desc', n_bars=None, interpolate_period=False, 
                          steps_per_period=10, compute_ranks=True):
        return pwd(self._df, orientation, sort, n_bars, interpolate_period,
                                 steps_per_period, compute_ranks)

    def prepare_long_data(self, index, columns, values, aggfunc='sum', orientation='h', 
                          sort='desc', n_bars=None, interpolate_period=False, 
                          steps_per_period=10, compute_ranks=True):
        return pld(self._df, index, columns, values, aggfunc, orientation, 
                   sort, n_bars, interpolate_period, steps_per_period, compute_ranks)


_BCR.bar_chart_race.__doc__ = re.sub('df : .*(?=filename :)', '',  bcr.__doc__, flags=re.S)
_BCR.bar_chart_race_plotly.__doc__ = re.sub('df : .*(?=filename :)', '',  bcrp.__doc__, flags=re.S)
_BCR.prepare_wide_data.__doc__ = re.sub('df : .*(?=filename :)', '',  pwd.__doc__, flags=re.S)
_BCR.prepare_long_data.__doc__ = re.sub('df : .*(?=filename :)', '',  pld.__doc__, flags=re.S)
