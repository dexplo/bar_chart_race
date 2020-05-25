import bar_chart_race as bcr


class TestSimpleBC:

    df = bcr.load_dataset('covid19')

    def test_defaults(self):
        bcr.bar_chart_race(self.df.iloc[-20:-16])
        bcr.bar_chart_race(self.df.iloc[-20:-16], orientation='v')

    def test_sort(self):
        bcr.bar_chart_race(self.df.iloc[-20:-16], sort='asc')
        bcr.bar_chart_race(self.df.iloc[-20:-16], orientation='v', sort='asc')

    def test_nbars(self):
        bcr.bar_chart_race(self.df.iloc[-20:-16], sort='desc', n_bars=8)
        bcr.bar_chart_race(self.df.iloc[-20:-16], orientation='v', sort='desc', n_bars=8)

    def fixed_order(self):
        bcr.bar_chart_race(df.iloc[-20:-16], sort='asc', n_bars=8, fixed_order=True)
        bcr.bar_chart_race(df.iloc[-20:-16], sort='asc', fixed_order=['Iran', 'USA', 'Italy', 'Spain'])