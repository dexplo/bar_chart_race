from ._bar_chart_race import bar_chart_race
import importlib
if importlib.util.find_spec('plotly'):
    from ._bar_chart_race_plotly import bar_chart_race_plotly
from ._line_chart_race import line_chart_race
from ._utils import load_dataset, prepare_wide_data, prepare_long_data
from . import _pandas_accessor

__version__ = '0.2.0'
__all__ = [
    'bar_chart_race', 
    'bar_chart_race_plotly',
    'load_dataset', 
    'prepare_wide_data', 
    'prepare_long_data',
    'line_chart_race']
