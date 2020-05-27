# Tutorial

bar_chart_race offers a wide range of inputs to customize the animation. On this page, we'll cover many of the available options.

## Data format

The data you choose to animate as a bar chart race must be provided in a specific format. The data must be within a pandas DataFrame containing 'wide' data where:

* Each row represents a single period of time
* Each column holds the value for a particular category
* The index contains the time component (optional)

### Example data

The data below is an example of properly formatted data. It shows total deaths from COVID-19 for several countries by date. Each row represents a single day's worth of data. Each column represents a single country's deaths. The index contains the date. Any pandas DataFrame that conforms to this structure may be used to create a bar chart race.

![](images/wide_data.png)

## Basic bar chart races

A single main function, `bar_chart_race`, exists to create the animations. Calling it with the defaults returns the animation as an HTML string. The `load_dataset` function is available to load sample DataFrames. If you are working within a Jupyter Notebook, you can import the `HTML` function from the `IPython.display` module to embed the video directly into the notebook.

```python
import bar_chart_race as bcr
from IPython.display import HTML
df = bcr.load_dataset('covid19')
html_string = bcr.bar_chart_race(df)
HTML(html_string)
```

<video controls><source src="videos/basic_1.mp4" type="video/mp4"></video>