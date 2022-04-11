
libratools is a Python library to process animal movement trajectories in the form of a series of locations (as x, y coordinates) with times. It is designed primarily for quantitative biology research projects that involve processing high-throughput tracking data.

libratools does not provide functionality to label trajectories; it operates on existing trajectories which are sequences of (x, y, time) coordinates. It can be used with any x, y, times data but is specifically built for trajectories collected using the [loopbio motif video recording system](http://loopbio.com/recording/) and labelled with [BioTracker](https://github.com/BioroboticsLab/biotracker_core), a computer vision framework for visual animal tracking. It does, however, provide some functionality to generate random trajectories for simulation or testing of analyses.

## Installation and Setup

The package can be installed from GitHub using:

 $ `pip install -e . --user.` 
 
 ## How to use libratools 

libratools is itself made up of several modules which each contain specific functions to handle tasks associated with loading, pre-processing, and post-processing trajectories. These can be contained in one or more scripts to handle pre-processing and post-processing tasks. For a detailed overview of each function, see the module's subpage.


## Questions or suggestions?

libratools is an open-sourced project, we welcome users to reach out with suggestions to improve the source code and encourage contributions.

Please read [CONTRIBUTING.md](https://github.com/vincejstraub/developing-exploration-behavior/blob/master/.github/CONTRIBUTING.md) for details on code conventions, and the process for submitting changes via pull requests. For major changes, please open an issue first to discuss what you would like to change.

Other enquiries about contributing and general questions can be sent to: vincejstraub@gmail.com
