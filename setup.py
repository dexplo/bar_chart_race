import setuptools
import re

with open('bar_chart_race/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split("'")[1]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bar_chart_race",
    version=version,
    author="Ted Petrou",
    author_email="petrou.theodore@gmail.com",
    description="Create animated bar chart races using matplotlib or plotly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="visualization animation bar chart race matplotlib pandas plotly",
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