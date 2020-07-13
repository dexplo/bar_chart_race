import re

import pandas as pd

from ._make_chart import bar_chart_race


@pd.api.extensions.register_dataframe_accessor("bcr")
class _BCR:

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def bar_chart_race(self, filename=None, orientation='h', sort='desc', n_bars=None, 
                       fixed_order=False, fixed_max=False, steps_per_period=10, 
                       period_length=500, interpolate_period=False, 
                       bar_label_position='outside', bar_label_fmt='{x:,.0f}',
                       bar_size=.95, period_label=True, period_fmt=None, 
                       period_summary_func=None, perpendicular_bar_func=None, 
                       figsize=(6, 3.5), cmap=None, title=None, bar_label_size=7, 
                       tick_label_size=7, shared_fontdict=None, scale='linear', writer=None, 
                       fig=None, dpi=144, bar_kwargs=None, filter_column_colors=False):

        return bar_chart_race(self._obj, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                            steps_per_period, period_length, interpolate_period, bar_label_position, 
                            bar_label_fmt, bar_size, period_label, period_fmt, period_summary_func, 
                            perpendicular_bar_func, figsize, cmap, title, bar_label_size, 
                            tick_label_size, shared_fontdict, scale, writer, fig, dpi, bar_kwargs, 
                            filter_column_colors)

doc = bar_chart_race.__doc__
_BCR.bar_chart_race.__doc__ = re.sub('df : .*(?=filename :)', '',  doc, flags=re.S)
