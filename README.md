# libratools  

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

<!---
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
--->

libratools is a Python library to process animal movement trajectories in the form of a series of locations (as x, y coordinates) with times. It is designed primarily for quantitative biology research projects that involve processing high-throughput tracking data.

libratools does not provide functionality to label trajectories; it operates on existing trajectories which are sequences of (x, y, time) coordinates. It can be used with any x, y, times data but is specifically built for trajectories collected using the [loopbio motif video recording system](http://loopbio.com/recording/) and labelled with [BioTracker](https://github.com/BioroboticsLab/biotracker_core), a computer vision framework for visual animal tracking. It does, however, provide some functionality to generate random trajectories for simulation or testing of analyses.

## Installation and Setup

 $ `pip install -e . --user.` 

## Documentation

The package functions are conveniently documented at the package website: https://vincejstraub.github.io/tools-libratools/.

For more information see the project [Wiki](https://github.com/vincejstraub/tools-libratools/wiki). 

## License

This project is licensed under a MIT License; see [LICENSE](https://github.com/vincejstraub/tools-libratools/blob/main/LICENSE) for details. By downloading libratools you agree with the following points: libratools is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of libratools.

## Contributing

Please read ____ for details on code conventions, and the process for submitting changes via pull requests.

## Maintenance

* [Vincent J. Straub](https://github.com/vincejstraub)  

## Requirements

Requirements are listed in the file :

* `requirements.txt`

Please follow  online instructions to install the required libraries, depending on your operating system and machine specifications.
