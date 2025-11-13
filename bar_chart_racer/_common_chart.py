import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker


class CommonChart:
    """Base class for shared chart functionality between matplotlib and plotly implementations."""

    def get_extension(self):
        """Extract file extension from filename."""
        if self.filename:
            return self.filename.split('.')[-1]

    def validate_common_params(self):
        """Validate parameters common to both matplotlib and plotly."""
        if isinstance(self.filename, str):
            if '.' not in self.filename:
                raise ValueError('`filename` must have an extension')
        elif self.filename is not None:
            raise TypeError('`filename` must be None or a string')

        if self.sort not in ('asc', 'desc'):
            raise ValueError('`sort` must be "asc" or "desc"')

        if self.orientation not in ('h', 'v'):
            raise ValueError('`orientation` must be "h" or "v"')

    def get_title(self, title):
        """Process title parameter for matplotlib."""
        if isinstance(title, str):
            return {'label': title}
        elif isinstance(title, dict):
            if 'label' not in title:
                raise ValueError('You must use the key "label" in the `title` dictionary '
                                 'to supply the name of the title')
        elif title is not None:
            raise TypeError('`title` must be either a string or dictionary')
        else:
            return {'label': None}
        return title

    def get_tick_template(self, tick_template):
        if isinstance(tick_template, str):
            return ticker.StrMethodFormatter(tick_template)
        elif callable(tick_template):
            return ticker.FuncFormatter(tick_template)

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
                                    "'cursive', 'family', 'fantasy', 'monospace', 'sans-serif',"
                                    "'serif', 'stretch','style', 'variant', 'weight'")
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

    def get_fig_kwargs(self, fig_kwargs):
        default_fig_kwargs = {'figsize': (6, 3.5), 'dpi': 144}
        if fig_kwargs is None:
            return default_fig_kwargs
        if isinstance(fig_kwargs, dict):
            fig_kwargs = {**default_fig_kwargs, **fig_kwargs}
        else:
            raise TypeError('fig_kwargs must be a dict or None')
        return fig_kwargs

    def get_fig(self, fig):
        if fig is not None and not isinstance(fig, plt.Figure):
            raise TypeError('`fig` must be a matplotlib Figure instance')
        if fig is not None:
            if not fig.axes:
                raise ValueError('The figure passed to `fig` must have an axes')
            ax = fig.axes[0]
        else:
            fig = self.create_figure()
        return fig

    def prepare_fixed_order_data(self, df):
        """Prepare data when fixed_order is True or a list."""
        if self.fixed_order is True:
            last_values = df.iloc[-1].sort_values(ascending=False)
            cols = last_values.iloc[:self.n_bars].index
            df = df[cols]
        elif isinstance(self.fixed_order, list):
            cols = self.fixed_order
            df = df[cols]
            self.n_bars = min(len(cols), self.n_bars)
        return df

    def create_fixed_order_ranks(self, df_values, cols):
        """Create rank DataFrame for fixed order data."""
        n = df_values.shape[1] + 1
        m = df_values.shape[0]
        rank_row = np.arange(1, n)
        if (self.sort == 'desc' and self.orientation == 'h') or \
           (self.sort == 'asc' and self.orientation == 'v'):
            rank_row = rank_row[::-1]

        ranks_arr = np.repeat(rank_row.reshape(1, -1), m, axis=0)
        return pd.DataFrame(data=ranks_arr, columns=cols)

    def get_col_filt(self):
        """Filter columns based on whether they appear in the animation."""
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

    def get_default_colormap(self):
        """Get default colormap based on number of columns."""
        if self.df_values.shape[1] > 10:
            return 'dark24'
        return 'dark12'

    def load_colormap(self, colors):
        """Load colormap from string name."""
        from ._colormaps import colormaps
        try:
            return colormaps[colors.lower()]
        except KeyError:
            raise KeyError(f'Colormap {colors} does not exist. Here are the '
                          f'possible colormaps: {list(colormaps.keys())}')

    def expand_colors_to_match_columns(self, bar_colors):
        """Expand color list to match number of columns."""
        n = len(bar_colors)
        if self.df_values.shape[1] > n:
            bar_colors = bar_colors * (self.df_values.shape[1] // n + 1)
        return np.array(bar_colors[:self.df_values.shape[1]])

    def warn_if_color_repetition(self, col_filt, n_colors):
        """Warn user if some columns never appear causing color repetition."""
        if not self.filter_column_colors and not col_filt.all():
            col_idx = np.where(col_filt)[0] % n_colors
            col_idx_ct = np.bincount(col_idx, minlength=n_colors)
            num_cols = max(col_filt.sum(), n_colors)
            exp_ct = np.bincount(np.arange(num_cols) % n_colors, minlength=n_colors)
            if (col_idx_ct > exp_ct).any():
                warnings.warn(
                    "Some of your columns never make an appearance in the animation. "
                    "To reduce color repetition, set `filter_column_colors` to `True`"
                )
