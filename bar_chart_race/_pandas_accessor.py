import re

import pandas as pd

from ._bar_chart_race import bar_chart_race as bcr
from ._line_chart_race import line_chart_race as lcr
from ._utils import prepare_wide_data as pwd, prepare_long_data as pld


@pd.api.extensions.register_dataframe_accessor("bcr")
class _BCR:

    def __init__(self, df):
        self._df = df

    def bar_chart_race(self, filename=None, orientation='h', sort='desc', n_bars=None, 
                   fixed_order=False, fixed_max=False, steps_per_period=10, 
                   period_length=500, end_period_pause=0, interpolate_period=False, 
                   period_label=True, period_template=None, period_summary_func=None,
                   perpendicular_bar_func=None, colors=None, title=None, bar_size=.95,
                   bar_textposition='outside', bar_texttemplate='{x:,.0f}',
                   bar_label_font=None, tick_label_font=None, tick_template='{x:,.0f}',
                   shared_fontdict=None, scale='linear', fig=None, writer=None, 
                   bar_kwargs=None,  fig_kwargs=None, filter_column_colors=False):

        return bcr(self._df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                        steps_per_period, period_length, end_period_pause, interpolate_period, 
                        period_label, period_template, period_summary_func, perpendicular_bar_func,
                        colors, title, bar_size, bar_textposition, bar_texttemplate, 
                        bar_label_font, tick_label_font, tick_template, shared_fontdict, scale, 
                        fig, writer, bar_kwargs, fig_kwargs, filter_column_colors)

    def line_chart_race(self, filename=None, n_lines=None, steps_per_period=10, 
                        period_length=500, end_period_pause=0, period_summary_func=None, 
                        line_width_data=None, agg_line_func=None, agg_line_kwargs=None, 
                        others_line_func=None, others_line_kwargs=None, fade=1, min_fade=.3, 
                        images=None, colors=None, title=None, line_label_font=None, 
                        tick_label_font=None, tick_template='{x:,.0f}', shared_fontdict=None, 
                        scale='linear', fig=None, writer=None, line_kwargs=None, 
                        fig_kwargs=None):
        return lcr(self._df, filename, n_lines, steps_per_period, period_length, end_period_pause, 
                    period_summary_func, line_width_data, agg_line_func, agg_line_kwargs, 
                    others_line_func, others_line_kwargs, fade, min_fade, images, colors, 
                    title, line_label_font, tick_label_font, tick_template, shared_fontdict, 
                    scale, fig, writer, line_kwargs, fig_kwargs)

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
_BCR.line_chart_race.__doc__ = re.sub('df : .*(?=filename :)', '',  lcr.__doc__, flags=re.S)
_BCR.prepare_wide_data.__doc__ = re.sub('df : .*(?=filename :)', '',  pwd.__doc__, flags=re.S)
_BCR.prepare_long_data.__doc__ = re.sub('df : .*(?=filename :)', '',  pld.__doc__, flags=re.S)

import importlib
if importlib.util.find_spec('plotly'):
    from ._bar_chart_race_plotly import bar_chart_race_plotly as bcrp
    def bar_chart_race_plotly(self, filename=None, orientation='h', sort='desc', n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=False, 
                          period_label=True, period_template=None, period_summary_func=None, 
                          perpendicular_bar_func=None, colors=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, scale='linear', 
                          bar_kwargs=None, layout_kwargs=None, write_html_kwargs=None, 
                          filter_column_colors=False):

        return bcrp(self._df, filename, orientation, sort, n_bars, fixed_order, fixed_max,
                        steps_per_period, period_length, end_period_pause, interpolate_period, 
                        period_label, period_template, period_summary_func, perpendicular_bar_func, 
                        colors, title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                        tick_label_font, hovertemplate, slider, scale, bar_kwargs, layout_kwargs, 
                        write_html_kwargs, filter_column_colors)

    setattr(_BCR, 'bar_chart_race_plotly', bar_chart_race_plotly)
    _BCR.bar_chart_race_plotly.__doc__ = re.sub('df : .*(?=filename :)', '',  bcrp.__doc__, flags=re.S)