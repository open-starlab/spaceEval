# OpenSTARLab SpaceEval package
[![Documentation Status](https://readthedocs.org/projects/openstarlab/badge/?version=latest)](https://openstarlab.readthedocs.io/en/latest/?badge=latest)
[![dm](https://img.shields.io/pypi/dm/openstarlab-spaceEval)](https://pypi.org/project/openstarlab-spaceEval/)
[![ArXiv](https://img.shields.io/badge/ArXiv-2502.02785-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.02785)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/PnH2MDDaCf)

## Introduction
The OpenSTARLab SpaceEval package is designed to provide simple and efficient way to visualize and estimate space evaluation for different sports. This package supports the data preprocessed by the OpenSTARLab PreProcessing package.


This package is continuously evolving to support future OpenSTARLab projects. If you have any suggestions or encounter any bugs, please feel free to open an issue.


## Installation

- To install this package via PyPI
```
pip install openstarlab-spaceEval
```
- To install manually
```
git clone git@github.com:open-starlab/spaceEval.git
cd ./spaceEval
pip install -e .
```
<details>
<summary><b>Class Method for Basketball data</b></summary>

## Class Method for Basketball data

- To have the values of probability for all the court (input = one or more line of dataframe, output = .json)
```
.get_values(data, json_path=None)
```
- To visualize specific frame (input = one line of dataframe,  output = .png)
```
.plot_heat_map_frame(data, save_path_folder,
                     include_player_velocities = True, BID=True, colorbar = True, title=True)
```
- To visualize specific sequence, (input = more than one line of dataframe output = .mp4)
```
plot_heat_map_sequence(data, save_path_folder,
                       heatmap=True, EVENT=True, JERSEY=True, BID=False, axis=False, title=True)
```

</details>