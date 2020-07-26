# Installation

Install from pypi or conda-forge

* `pip install bar_chart_race`
* `conda install -c conda-forge bar_chart_race`

## Installing ffmpeg

In order to save animations as mp4/m4v/mov/etc... files, you must [install ffmpeg][0], which allows for conversion to many different formats of video and audio. For macOS users, installation may be [easier using Homebrew][2].

After installation, ensure that `ffmpeg` has been added to your path by going to your command line and entering `ffmepg -version`.

## Install ImageMagick for animated gifs

If you desire to create animated gifs, you'll need to [install ImageMagick][1]. Verify that it has been added to your path with `magick -version`.

## Dependencies

Bar Chart Race requires that you have both matplotlib and pandas installed.

[0]: https://www.ffmpeg.org/download.html
[1]: https://imagemagick.org/
[2]: https://trac.ffmpeg.org/wiki/CompilationGuide/macOS#ffmpegthroughHomebrew