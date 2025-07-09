# OpenSTARLab SpaceEval package

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

## Class Method

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