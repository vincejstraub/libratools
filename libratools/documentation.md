
libratools is a Python library to process animal movement trajectories in the form of a series of locations (as x, y coordinates) with times. It is designed primarily for quantitative biology research projects that involve processing high-throughput tracking data.

libratools does not provide functionality to label trajectories; it operates on existing trajectories which are sequences of (x, y, time) coordinates. It can be used with any x, y, times data but is specifically built for trajectories collected using the [loopbio motif video recording system](http://loopbio.com/recording/) and labelled with [BioTracker](https://github.com/BioroboticsLab/biotracker_core), a computer vision framework for visual animal tracking. It does, however, provide some functionality to generate random trajectories for simulation or testing of analyses.

## Installation and Setup

The package can be installed from GitHub using:

 $ `pip install -e . --user.` 
