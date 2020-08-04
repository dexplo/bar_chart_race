from pathlib import Path

import pandas as pd
from matplotlib import image as mimage


def load_dataset(name='covid19'):
    '''
    Return a pandas DataFrame suitable for immediate use in `bar_chart_race`.
    Must be connected to the internet

    Parameters
    ----------
    name : str, default 'covid19'
        Name of dataset to load from the bar_chart_race github repository.
        Choices include:
        * 'covid19'
        * 'covid19_tutorial'
        * 'urban_pop'
        * 'baseball'

    Returns
    -------
    pandas DataFrame
    '''
    url = f'https://raw.githubusercontent.com/dexplo/bar_chart_race/master/data/{name}.csv'

    index_dict = {'covid19_tutorial': 'date',
                  'covid19': 'date',
                  'urban_pop': 'year',
                  'baseball': None}
    index_col = index_dict[name]
    parse_dates = [index_col] if index_col else None
    return pd.read_csv(url, index_col=index_col, parse_dates=parse_dates)

def prepare_wide_data(df, orientation='h', sort='desc', n_bars=None, interpolate_period=False, 
                      steps_per_period=10, compute_ranks=True):
    '''
    Prepares 'wide' data for bar chart animation. 
    Returns two DataFrames - the interpolated values and the interpolated ranks
    
    There is no need to use this function directly to create the animation. 
    You can pass your DataFrame directly to `bar_chart_race`.

    This function is useful if you want to view the prepared data without 
    creating an animation.

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'wide' pandas DataFrame where each row represents a 
        single period of time. 
        Each column contains the values of the bars for that category. 
        Optionally, use the index to label each time period.

    orientation : 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort : 'desc' or 'asc', default 'desc'
        Choose how to sort the bars. Use 'desc' to put largest bars on 
        top and 'asc' to place largest bars on bottom.

    n_bars : int, default None
        Choose the maximum number of bars to display on the graph.
        By default, use all bars. New bars entering the race will 
        appear from the bottom or top.

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

    steps_per_period : int, default 10
        The number of steps to go from one time period to the next. 
        The bars will grow linearly between each period.

    compute_ranks : bool, default True
        When `True` return both the interpolated values and ranks DataFrames
        Otherwise just return the values

    Returns
    -------
    A tuple of DataFrames. The first is the interpolated values and the second
    is the interpolated ranks.

    Examples
    --------
    df_values, df_ranks = bcr.prepare_wide_data(df)
    '''
    if n_bars is None:
        n_bars = df.shape[1]

    df_values = df.reset_index()
    df_values.index = df_values.index * steps_per_period
    new_index = range(df_values.index[-1] + 1)
    df_values = df_values.reindex(new_index)
    if interpolate_period:
        if df_values.iloc[:, 0].dtype.kind == 'M':
            first, last = df_values.iloc[[0, -1], 0]
            dr = pd.date_range(first, last, periods=len(df_values))
            df_values.iloc[:, 0] = dr
        else:
            df_values.iloc[:, 0] = df_values.iloc[:, 0].interpolate()
    else:
        df_values.iloc[:, 0] = df_values.iloc[:, 0].fillna(method='ffill')
    
    df_values = df_values.set_index(df_values.columns[0])
    if compute_ranks:
        df_ranks = df_values.rank(axis=1, method='first', ascending=False).clip(upper=n_bars + 1)
        if (sort == 'desc' and orientation == 'h') or (sort == 'asc' and orientation == 'v'):
            df_ranks = n_bars + 1 - df_ranks
        df_ranks = df_ranks.interpolate()
    
    df_values = df_values.interpolate()
    if compute_ranks:
        return df_values, df_ranks
    return df_values

def prepare_long_data(df, index, columns, values, aggfunc='sum', orientation='h', 
                      sort='desc', n_bars=None, interpolate_period=False, 
                      steps_per_period=10, compute_ranks=True):
    '''
    Prepares 'long' data for bar chart animation. 
    Returns two DataFrames - the interpolated values and the interpolated ranks
    
    You (currently) cannot pass long data to `bar_chart_race` directly. Use this function
    to create your wide data first before passing it to `bar_chart_race`.

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'long' pandas DataFrame where one column contains 
        the period, another the categories, and the third the values 
        of each category for each period. 
        
        This DataFrame will be passed to the `pivot_table` method to turn 
        it into a wide DataFrame. It will then be passed to the 
        `prepare_wide_data` function.

    index : str
        Name of column used for the time period. It will be placed in the index

    columns : str
        Name of column containing the categories for each time period. This column
        will get pivoted so that each unique value is a column.

    values : str
        Name of column holding the values for each time period of each category.
        This column will become the values of the resulting DataFrame

    aggfunc : str or aggregation function, default 'sum'
        String name of aggregation function ('sum', 'min', 'mean', 'max, etc...) 
        or actual function (np.sum, np.min, etc...). 
        Categories that have multiple values for the same time period must be 
        aggregated for the animation to work.

    orientation : 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort : 'desc' or 'asc', default 'desc'
        Choose how to sort the bars. Use 'desc' to put largest bars on 
        top and 'asc' to place largest bars on bottom.

    n_bars : int, default None
        Choose the maximum number of bars to display on the graph.
        By default, use all bars. New bars entering the race will 
        appear from the bottom or top.

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

    steps_per_period : int, default 10
        The number of steps to go from one time period to the next. 
        The bars will grow linearly between each period.

    compute_ranks : bool, default True
        When `True` return both the interpolated values and ranks DataFrames
        Otherwise just return the values

    Returns
    -------
    A tuple of DataFrames. The first is the interpolated values and the second
    is the interpolated ranks.

    Examples
    --------
    df_values, df_ranks = bcr.prepare_long_data(df)
    bcr.bar_chart_race(df_values, steps_per_period=1, period_length=50)
    '''
    df_wide = df.pivot_table(index=index, columns=columns, values=values, 
                             aggfunc=aggfunc).fillna(method='ffill')
    return prepare_wide_data(df_wide, orientation, sort, n_bars, interpolate_period,
                             steps_per_period, compute_ranks)


def read_images(filename, columns):
    image_dict = {}
    code_path = Path(__file__).resolve().parent / "_codes"
    code_value_path = code_path / 'code_value.csv'
    data_path = code_path / f'{filename}.csv'
    url_path = pd.read_csv(code_value_path).query('code == @filename')['value'].values[0]
    codes = pd.read_csv(data_path, index_col='code')['value'].to_dict()

    for col in columns:
        code = codes[col.lower()]
        if url_path == 'self':
            final_url = code
        else:
            final_url = url_path.format(code=code)
        image_dict[col] = mimage.imread(final_url)
    return image_dict