import setuptools
import re
from bar_chart_race import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

pat = r'!\[img\]\('
repl = r'![img](https://raw.githubusercontent.com/dexplo/bar_chart_race/master/'
long_description = re.sub(pat, repl, long_description)

setuptools.setup(
    name="bar_chart_race",
    version=__version__,
    author="Ted Petrou",
    author_email="petrou.theodore@gmail.com",
    description="Create animated bar chart races using matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="visualization animation bar chart race matplotlib pandas",
    url="https://github.com/dexplo/bar_chart_race",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas>=0.24", "matplotlib>=3.1"],
    python_requires='>=3.6',
    include_package_data=True,
)