#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_inspect module includes functions for inspecting a trajectory
dataset.
"""

import datetime     # standard library

import numpy as np     # 3rd party packages
import pandas as pd

from . import lbt_utils   # local imports
from . import lbt_metrics
from . import lbt_datasets


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


def main():
    pass


def get_step_len_stats(trajectory, step_col='stepLength'):
    """
    Returns mean, max, var, and standard deviation for step length,
    where step refers to euclidean distance between two consecutive
    data points in 2 dimension using the time interval between frame
    values set in BioTracker.

    Args:
        trajectory (dict): dictionary of trajectory DataFrame and metadata.
        step_col (str, default='stepLength'): DataFrame column containg step
            lengths.
            
    Returns:
        A dict of mean, max, var and std.
    """
    # append step length between frames to column
    if step_col not in trajectory.columns:
        trajectory[step_col] = lbt_metrics.get_step_len(
            trajectory['x'], trajectory['y'])
        
    # get euclidean distance values
    step_vals = trajectory[step_col]

    stats = {}
    # compute mean, max, min, var, and stdev
    stats['mean_step_len'] = np.mean(step_vals)
    stats['max_step_len'] = np.max(step_vals)
    stats['step_len_var'] = np.var(step_vals)
    stats['step_len_std'] = np.std(step_vals)

    return stats


def summarize(trajectory, x, y, time_col='timestamp', freq='60min',
              unit='hr', FPS=5, display=False):
    """
    Returns summary statistics for primary movement metrics activity
    (displacement), cumulative step leangth between consecutive coordinate
    points, and relative turning angle: activity, activity per interval,
    mean interval activity, median interval activity, mean step len, max step
    len, step len variance, step len standard deviation, mean turning angle,
    turning angle standard deviation.

    Args:
        trajectory (dict): dictionary of trajectory DataFrame and metadata.
        x (list or series): x-coordinate data points for distance calculation.
        y (list of series): y-coordiante data points for distance calculation.
        time_col (str, default='timestamp'): column to groupby for average
            distance calculations per trajectory segment, must be datetime.
        freq (str, default='60min'): length of time interval for which to
            compute interval summary statistics.
        unit (str, default='hr'): length of time interval for which to 
            compute mean activity, must be one of: sec, min, hr.
        FPS (int, default=5): frames per second.
        display(bool, default=False): if display=True, dictionary of summary
            statistics is printed out.

    Returns:
        stats (dict): dictionary of summary statistics.
    """
    stats = {}

    # group by datetime column
    trajectory[time_col] = pd.to_datetime(trajectory[time_col])

    # compute total activity (cumulative euclidean distance)
    total_activity = np.sum(lbt_metrics.get_step_len(x, y))
    stats['activity'] = total_activity

    # group by time interval
    ti = trajectory.groupby([pd.Grouper(key=time_col, freq=freq)])
    # compute displacement for each time interval
    interval_activity_vals = {k: np.sum(lbt_metrics.get_step_len(
        ti.get_group(k)['x'], ti.get_group(k)['y'])
                                   ) for k, v in ti.groups.items()}
    # append to dict
    stats['interval_activity_vals'] = interval_activity_vals
    # compute median activity per time interval
    stats['med_interval_activity'] = np.nanmedian(
        list(interval_activity_vals.values()))

    # append mean activity per unit time 
    mean_activity = total_activity / len(trajectory)
    if unit == 'sec':
        mean_activity_per_unit_time = mean_activity * FPS
    elif unit == 'min':
        mean_activity_per_unit_time = mean_activity * (60 * FPS)
    elif unit == 'hr':
        mean_activity_per_unit_time = mean_activity * (3600 * FPS)
    # append mean value to dict
    stats['mean_activity'] = mean_activity_per_unit_time
    
    # compute step length statistics
    step_len_stats = get_step_len_stats(trajectory)
    stats.update(step_len_stats)

    # compute relative turning angles across segments
    turning_angles = lbt_metrics.get_relative_turn_angle(trajectory)
    # append mean and standard deviation of distribution
    stats['mean_turning_angle'] = np.nanmean(turning_angles)
    stats['turning_angle_std'] = np.nanstd(turning_angles)

    if display is True:
        print(stats)

    return stats


def get_track_len(file_paths):
    """
    Returns total track length (amount of recording footage) as string,
    in HH:MM:SS format, across all supplied CSV files, number of file
    paths provided rounded up.

    Args:
        file_paths (list): list of CSV file paths as strings.
    """
    num_files = len(file_paths)
    # count total track length for each .csv file
    track_lens_ms = []
    for path in file_paths:
        # load file into DataFrame
        df, _ = lbt_datasets.load_csv(path, na_summary=False,
                                      warn_bad_lines=False)
        try:
            # get total time in milliseoconds
            track_len_ms = df['MillisecsByFPS'].tail(1).item()
        except ValueError:
            print('Error when trying to read:\n{}'.format(path))
        # store track length in ms and append to list
        track_lens_ms.append(track_len_ms)

    # sum and convert to seconds
    total_track_len_ms = sum(track_lens_ms)
    total_track_len_s = int(lbt_utils.convert_milliseconds(
        total_track_len_ms)['secs'])

    # store as string in datetime format
    time = datetime.timedelta(seconds=total_track_len_s)

    return time, num_files


def get_recording_time_elapsed(df, num_missing_vals, time_col='MillisecsByFPS',
                     time_interval=40, unit='mins'):
    """
    Returns maximum value of timekeeping column in a DataFrame after
    subtracting amount of time missed as a result of NaN values.

    Args:
        df (pandas DataFrame): pandas DataFrame.
        num_missing_vals (int): number of NaNs in key columns
        time_col (str, default=MillisecsByFPS): time-keeping column.
        time_interval (int, default=40): time interval between rows
            in milliseconds.
        unit (str, default=hours): time unit for returning time_elapsed.

    Returns:
        time_elapsed (int)
    """
    total_time_elapsed = df[time_col].iloc[-1]
    time_missing = num_missing_vals * time_interval
    time_elapsed = total_time_elapsed - time_missing
    if unit in ['secs', 'mins', 'hrs', 'days']:
        time_elapsed = lbt_utils.convert_milliseconds(time_elapsed)[unit]
    elif unit == 'ms':
        pass
    else:
        raise Exception("Unit is not one of: ms, secs, hrs, days.")

    return time_elapsed


def time_change_from_frame_count(frame_count, time_between_frames=0.2,
                                 unit='mins'):
    """
    Returns time elapsed based on frame count and time between frames.
    
    Args:
        frame_count (int): number of frames.
        time_between_frames (float, default=0.2): time between frames in 
            seconds.
        unit (str, default=mins): time unit for returning time_elapsed
        
    Returns:
        time_elapased (int)
    """
    d_s = frame_count * time_between_frames
    d_mins, d_hrs = lbt_utils.convert_seconds(d_s)
    if unit == 'mins':
        return d_mins
    elif unit == 'hrs':
        return d_hrs
    else:
        print('unit needs to be one of: mins, hrs.')
    

def time_change_between_timestamps(starttime, endtime, 
                                   str_format='%Y-%m-%d %H:%M:%S', unit='mins'):
    """
    Returns time elapsed between two datetime timestamp values.
    
    Args:
        starttime (datetime.datetime): datetime object corresponding to 
            start time. 
        endtime (datetime.datetime): datetime object corresponding to end
            time from which to subtract starttime.
        str_format (str, default='%Y-%m-%d %H:%M:%S'): time format of str if
            strattime or endtime are str objects that need to be converted to 
            datetime objects. 
        unit (str, default=mins): time unit for returning time_elapsed
        
    Returns:
        time_elapased (datetime.datetime)
    """
    if type(starttime) == str:
        starttime = datetime.strptime(starttime, str_format)
    if type(endtime) == str:
        endtime = datetime.strptime(endtime, str_format)
    d_s = (endtime - starttime).total_seconds()
    d_mins, d_hrs = lbt_utils.convert_seconds(d_s)
    if unit == 'mins':
        return d_mins
    elif unit == 'hrs':
        return d_hrs
    else:
        print('unit needs to be one of: mins, hrs.')
        
        
if __name__ == '__main__':
    main()
