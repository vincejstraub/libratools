#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_metrics module includes performance metrics, pairwise metrics
and distance computations to summarize a dataset.
"""


import numpy as np    # 3rd party packages
import pandas as pd

from . import lbt_utils    # local imports


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


def get_step_len(x, y):
    """
    Returns step length (size), the displacement between two consecutive
    coordinate points, calculates as stepwise euclidean distance using 
    the formula:

    res = ∑i=1,...,n-1 sqrt[ (xi - xi-1)2 + (yi - yi-1)2 ]

    Args:
        x (list or series): list of values in first dimension.
        y (list or series): list of values in second dimension.
    """
    # convert arguments to arrays
    x, y = list(x), list(y)
    xy = np.array(list(zip(x, y)))

    # find the differences
    diff = np.diff(xy, axis=0)

    # raise to the power of 2 and sum
    ss = np.power(diff, 2).sum(axis=1)

    # find the square root
    res = np.sqrt(ss)

    # prepend 0 to the first value
    res = np.insert(res, 0, 0.0, axis=0)

    return res


def get_relative_turn_angle(trajectory, heading_col='heading'):
    """
    Computes step angles (in degrees) of each location relative to the 
    previous location. Angles may be between -180 and 180 degrees where 
    negative degrees indicate 'left' turns. Users should apply this 
    function to one trajectory at a time.

    Args:
        trajectory (pandas.DataFrame): a DataFrame which must have the columns
            specified by x, y, and time.
        heading_col (tr, default='heading'): DataFrame column containing 
            trajectory headings. 

    Returns:
        angles (pandas.Series): a vector of turning angles in degrees. There are
            two fewer angles than the number of rows; these are filled with NaN
            values.
    """
    if heading_col not in trajectory:
        heading = get_heading(trajectory)
    else:
        heading = trajectory.heading
    
    turn_angle = heading.diff().rename("turnAngle")
    
    # correction for 360-degree angle range
    turn_angle.loc[turn_angle >= 180] -= 360
    turn_angle.loc[turn_angle < -180] += 360
    
    return turn_angle


def get_heading(trajectory, angle_col='angle'):
    """
    Compute trajectory heading.
        trajectory (pandas.DataFrame): a DataFrame which must have the columns
            specified by x, y, and time.
        heading_col (tr, default='heading'): DataFrame column containing 
            trajectory headings. 

    Returns:
        angles (pandas.Series): a vector of turning angles in degrees. There are
            two fewer angles than the number of rows; these are filled with NaN
            values.
    """
    # check step length exists as column
    if angle_col not in trajectory.columns:
        angle = get_absoloute_turn_angle(trajectory)
    else:
        angle = trajectory.angle

    dx = trajectory.x.diff()
    dy = trajectory.y.diff()
    
    # get heading from angle
    mask = (dx > 0) & (dy >= 0)
    trajectory.loc[mask, "heading"] = angle[mask]
    mask = (dx >= 0) & (dy < 0)
    trajectory.loc[mask, "heading"] = -angle[mask]
    mask = (dx < 0) & (dy <= 0)
    trajectory.loc[mask, "heading"] = -(180 - angle[mask])
    mask = (dx <= 0) & (dy > 0)
    trajectory.loc[mask, "heading"] = 180 - angle[mask]
    
    return trajectory.heading


def get_absoloute_turn_angle(trajectory, step_col='stepLength', unit="degrees", lag=1):
    """
    Returns angle between steps as a function of displacement with regard to x axis.

    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        unit (str, default='degrees'): unit to compute angles in, set to 
            degrees by default.
        lag (int): angles between every lag'th element are calculated.
    
    Returns:
        angles (pandas.Series): a vector of turning angles in degrees. There are
            two fewer angles than the number of rows; these are filled with NaN
            values.
    """
    # check step length exists as column
    if step_col not in trajectory.columns:
        trajectory[step_col] = lbt_metrics.get_step_len(
            trajectory['x'], trajectory['y'])

    if unit == "degrees":
        angle = np.rad2deg(np.arccos(np.abs(trajectory.x.diff(lag)) / trajectory[step_col]))
    elif unit == "radians":
        angle = np.arccos(np.abs(trajectory.x.diff()) / trajectory[step_col])
    else:
        raise ValueError(f"The unit {unit} is not valid, expected radians or degrees.")

    angle.unit = unit
    angle.name = "angle"
    
    return angle
