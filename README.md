# Bar Chart Race
Make animated bar and line chart races in Python with matplotlib or plotly.

Original Repo (without icons) : [https://github.com/dexplo/bar_chart_race](https://github.com/dexplo/bar_chart_race)

Andres Berejnoi's Repo (with icons) : [https://github.com/andresberejnoi/bar_chart_race](https://github.com/andresberejnoi/bar_chart_race)

---

## Popular Programming Languages Bar Chart Race
![img](programming_languages.gif)

---

## Installation

Install using `pip`:

```
pip install git+https://github.com/programiz/bar_chart_race.git@master
```

You can also clone this repository and modify it as per your needs:

```
git clone https://github.com/programiz/bar_chart_race.git
```

---

## Usage

Create a file and use the `bar_chart_race` library as shown below:

```python
import pandas as pd
import bar_chart_race as bcr

df = pd.read_csv("data/language-popularity.csv", index_col='Date')
# replace empty values with 0
df.fillna(0.0, inplace=True)

# plotting the graph
bcr.bar_chart_race(
    df=df,
    filename='video.mp4',
    orientation='h',
    sort='desc',
    n_bars=10,
    fixed_order=False,
    fixed_max=False,
    steps_per_period=45,  # smoothness
    interpolate_period=False,
    # label_bars=True,
    bar_size=.95,
    period_label={'x': .95, 'y': .15,
                  'ha': 'right',
                  'va': 'center',
                  'size': 72,
                  'weight': 'semibold'
                  },

    shared_fontdict={'family': 'Euclid Circular A',
                     'weight': 'medium', 'color': '#25265E'},
    perpendicular_bar_func=None,
    period_length=1500,  # time period in ms per data
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .99, 'lw': 0},
    filter_column_colors=True,
    # change
    fig_kwargs={'figsize': (26.67, 15), 'dpi': 144, 'facecolor': '#F8FAFF'},
    colors='brand_colors',
    title={'label': 'Programming Language Popularity 1990 - 2020',
           'size': 52,
           'weight': 'bold',
           # 'loc': 'right',
           'pad': 40},
    bar_label_font={'size': 27},  # bar text size
    tick_label_font={'size': 27},  # y-axis text size
    img_label_folder='bar_image_labels',
)
```