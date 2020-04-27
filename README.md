# Bar Chart Race

Make animated bar chart races with matplotlib.

![][0]

## Usage

Must use a pandas DataFrame containing 'wide' data where:

* Every row represents a single period of time
* Each column holds the value for a particular category
* The index contains the time component (optional)
  
The data below is an example of properly formatted data. It shows total deaths from COVID-19 for the highest 20 countries by date.

| date                |   Belgium |   Brazil |   Canada |   China |   France |   Germany |   India |   Indonesia |   Iran |   Ireland |   Italy |   Mexico |   Netherlands |   Portugal |   Spain |   Sweden |   Switzerland |   Turkey |   USA |   United Kingdom |
|:--------------------|----------:|---------:|---------:|--------:|---------:|----------:|--------:|------------:|-------:|----------:|--------:|---------:|--------------:|-----------:|--------:|---------:|--------------:|---------:|------:|-----------------:|
| 2020-04-18 |      5453 |     2354 |     1399 |    4636 |    19345 |      4459 |     521 |         535 |   5031 |       571 |   23227 |      546 |          3613 |        687 |   20043 |     1511 |          1368 |     1890 | 38671 |            15498 |
| 2020-04-19 |      5683 |     2462 |     1563 |    4636 |    19744 |      4586 |     559 |         582 |   5118 |       610 |   23660 |      650 |          3697 |        714 |   20453 |     1540 |          1393 |     2017 | 40664 |            16095 |
| 2020-04-20 |      5828 |     2587 |     1725 |    4636 |    20292 |      4862 |     592 |         590 |   5209 |       687 |   24114 |      686 |          3764 |        735 |   20852 |     1580 |          1429 |     2140 | 42097 |            16550 |
| 2020-04-21 |      5998 |     2741 |     1908 |    4636 |    20829 |      5033 |     645 |         616 |   5297 |       730 |   24648 |      712 |          3929 |        762 |   21282 |     1765 |          1478 |     2259 | 44447 |            17378 |
| 2020-04-22 |      6262 |     2906 |     2075 |    4636 |    21373 |      5279 |     681 |         635 |   5391 |       769 |   25085 |      857 |          4068 |        785 |   21717 |     1937 |          1509 |     2376 | 46628 |            18151 |

### Main function - `bar_chart_race`

Only one main function exists, **`bar_chart_race`** that saves the animation to disk.

```python
>>> import bar_chart_race as bcr
>>> df = bcr.load_dataset('covid19')
>>> brc.bar_chart_race(df, 'filename.mp4', steps_per_period=10, title='COVID-19 Deaths by Country')
```

[0]: videos/covid19_horiz_desc.gif