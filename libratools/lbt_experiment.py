#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_experiment module contains functions related to various
experimental procedures, such as defining and assigning a treatment value.
"""

import random    # standard library
import datetime


# def main():
#     pass


def get_feeding_times(activity_vals, min_time=2, max_time=180,
                      end_time='12:00:00', round_min=False):
    """
    Returns feeding times based on range of daily activity.

    Args:
        activity_vals (list): list of activity values.
        min_time (int, default=2): minimum feeding time in mins.
        max_time (int, default=180): maximum feeding time in mins.
        end_time (str, default='12:00:00') time at which to stop
            feeding, if end_time='12:00:00' then feeding times will
            be computed to end at 12:00:00.
        round_min (bool, default=False): decide whether to round
            begin treatment time to the nearest minute, if round_min=
            True then time will be provided in format HH:MM:SS.

    Returns:
        feeding_time (list): list of feeding times in mins as ints
        start_times (list): list of feeding start times as strings.
    """
    # get current date
    today = datetime.datetime.today().strftime('%b %m %Y')
    # declare end of feeding time as datetime object
    end = datetime.datetime.strptime(f'{today} {end_time}', '%b %m %Y %H:%M:%S')

    # get feeding time values as list of ints using linear mapping
    feeding_times = apply_linear_mapping(
        activity_vals, y_min=min_time, y_max=max_time)

    # get feeding start times as strings in 12-hour clock time format
    if round_min is True:
        time_format = '%H:%M'
    else:
        time_format = '%H:%M:%S'

    start_times = [(end-datetime.timedelta(minutes=t)).strftime(f'{time_format} %p')
                   for t in feeding_times]

    return feeding_times, start_times


def apply_linear_mapping(x_set, y_min, y_max):
    """
    Applies a straight line equation in the form y = mx + b to a set of x
    values. First computes the slope-intercept form of the equation using
    the minimum and maximum of the provided x values alongside given minimum
    and maximum y values as points and then inputs each x value to the
    equation.

    Args:
        x_set (list): list of x coordinates.
        y_min (int): minimum y coordinate.
        y_max (int): maximum y coordinate.

    Returns:
        y_vals (list).
    """
    # declare range of y values
    y_set = list(range(y_min, y_max+1))

    # get constants for treatment equation
    m, b = get_slope_intercept_vars(
        (min(x_set), min(y_set)), (max(x_set), max(y_set)))

    # get y value for each activity value and return as list
    y_vals = [(m * x) + b for x in x_set]

    return y_vals


def get_slope_intercept_vars(p1, p2):
    """
    Returns slope and y-intercept for a straight line given two points.

    Args:
        p1 (tuple): first point.
        p2 (tuple): second point.

    Returns:
        m (float), b (float)
    """
    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    except ZeroDivisionError:
        m = (p2[1] - p1[1]) / 1
    b = p2[1] - (m * p2[0])

    return m, b


# if __name__ == '__main__':
#     # generate 12 random values for daily activity
#     activity_vals = random.sample(range(1000, 20000), 12)
#     # compute feeding times and start times based activity using defaults
#     feeding_times, feeding_start_times = get_feeding_times(activity_vals)
#     # return feeding time dict
#     treatment_dict = {x: {str(t): str(start)} for x, t, start in zip(
#         activity_vals, feeding_times, feeding_start_times)
#                      }
#     # iterate over the list of dict(s)
#     print('Activity (cm) | Feeding time | Begin treatment at}')
#     print('------------------------------------')
#     for val in treatment_dict:
#         print(val, treatment_dict[val])
