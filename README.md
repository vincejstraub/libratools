# libratools  

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

<!---
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
--->

libratools is a python package to process animal movement trajectories in the form of a series of locations (as x, y coordinates) with times. It is designed primarily for quantitative biology research projects that involve processing high-throughput tracking data.

libratools does not provide functionality to label trajectories; it operates on existing trajectories which are sequences of (x, y, time) coordinates. It can be used with any x, y, times data but is specifically built for trajectories collected using the [loopbio motif video recording system](http://loopbio.com/recording/) and labelled with [BioTracker](https://github.com/BioroboticsLab/biotracker_core), a computer vision framework for visual animal tracking. It does, however, provide some functionality to generate random trajectories for simulation or testing of analyses.

## Installation and Setup

 $ `pip install -e . --user.` 

## Usage

It takes as input the date (YYYYMMDD) and produces a dashboard as output for recordings for that day.

To preprocess files, run the following argument from the command line: 

`$ python preprocess.py date`

Or within a notebook:

```python
# To preprocess files, run the following argument from the command line: 
python preprocess.py date 
```

## User Agreement

By downloading libratools you agree with the following points: libratools is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of libratools.

## Maintenance

* [Vincent J. Straub](https://github.com/vincejstraub)  

## Requirements

Requirements are listed in the file :

* `requirements.txt`

Please follow  online instructions to install the required libraries, depending on your operating system and machine specifications.

## Acknowledgements

Parts of libratools' code implementation and analytical methods (particularly `get_relative_turning_angle()`) are inspired by Pratik Gupte's R package [`atlastools`](https://github.com/pratikunterwegs/atlastools). 
