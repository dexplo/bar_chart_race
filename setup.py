import setuptools
import re

with open('bar_chart_racer/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split("'")[1]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bar_chart_racer",
    version=version,
    author="Wibo van der Sluis",  # TODO: Update with your name
    author_email="wibo@sofus.nl",  # TODO: Update with your email
    description="Create animated bar chart races using matplotlib and plotly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="visualization animation bar chart race matplotlib pandas plotly",
    url="https://github.com/sofus-nl/bar_chart_racer",  # TODO: Update with your GitHub username
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    install_requires=[
        "pandas>=2.3.0",
        "matplotlib>=3.10.0",
        "plotly>=6.3.0",
    ],
    python_requires='>=3.11',
    include_package_data=True,
)