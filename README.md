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
    # Must be a 'wide' DataFrame where each row represents a single period of time.
    df=df,

    # name of the video file
    filename='video.mp4',

    # orientation of the bar
    orientation='h',

    # sort the bar for each period
    sort='desc',

    # number of bars to display on the video
    n_bars=10,

    # to fix the maximum value of the axis
    fixed_max=False,

    # smoothness of the animation
    steps_per_period=45,

    # time period in ms for each row
    period_length=1500,  

    # to adjust the postion and style of the custom label
    period_label={'x': .95, 'y': .15,
                  'ha': 'right',
                  'va': 'center',
                  'size': 72,
                  'weight': 'semibold'
                  },

    # set the color of the bars; pass a dictionary of hex values
    colors='brand_colors',

    # title and its styles
    title={'label': 'Programming Language Popularity 1990 - 2020',
           'size': 52,
           'weight': 'bold',
           # 'loc': 'right',
           'pad': 40},
           
    # adjust the width of each bar
    bar_size=.95,
    
    # style the bar label text
    bar_label_font={'size': 27},

    # stlye the y-axis texts (name of each bar)
    tick_label_font={'size': 27},  

    # adjust the sytle of bar
    # alpha for opacity of bar
    # ls - width of edge
    # ec - edgecolor
    bar_kwargs={'alpha': .99, 'lw': 0},

    # adjust the properties of video
    # figsize - resolution
    # facecolor - background color
    # dpi - dots per inch
    fig_kwargs={'figsize': (26.67, 15), 'dpi': 144, 'facecolor': '#F8FAFF'},

    # adjust the bar label format
    bar_texttemplate='{x:,.0f}',
    
    # adjust tick format
    tick_template='{x:,.0f}',

    # provide the name of folder that contains all the images
    img_label_folder='bar_image_labels',
)
```