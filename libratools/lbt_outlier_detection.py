#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_outlier_detection module includes methods for detecting
point and subsequence outliers.
"""

import numpy as np     # 3rd party packages
import pandas as pd

from . import lbt_utils    # local imports
from . import lbt_metrics


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


def run_detection(trajectory, step_col='stepLength', frame_col='globalFRAME',
                  segment_col='chunk_segment', x='x', y='y', thresh=5, fps=5, 
                  seconds=1, spike_method='interpolate', spike_seq_method='exclude',
                  corrupt_thresh=10):
    """
    Implements outlier detection by detecting subsequence outliers, point
    outliers and checking whether the movement track is corrupt, i.e. a 
    certain number of data points labelled as outliers exceed a used-defined
    data corruption threshold.
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        segment_col (str, default='chunk_segment'): DataFrame column 
            containing count of trajectory segment.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, deafult=5): Maximum step length between frames.
        fps (int, default=5): number of frames per second.
        seconds (int, default=1): number of seconds for which to exclude
            rows before commencing outlier detection.
        spike_method (str, default='interpolate'): method for handling spikes, 
            can be either 'interpolated' or 'keep'.
        spike_seq_method (str, default='drop'): method for handling spike 
            sequences, can be either 'interpolated' or 'keep'.
        corrupt_thresh (int, default=10): the number of data points labelled
            as outliers as a percentage of all data points that a track
            is allowed to have, if this value is exceeded the track is
            labelled as being likely corrupted.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """    
    # detect point outliers or 'spikes' and handle accordingly
    trajectory, spike_stats = detect_spikes(
        trajectory, step_col, frame_col, segment_col, x, y, thresh, 
        fps, seconds, spike_method)  
    
    # detect subsequence outliers or 'prolonged spikes' and handle accordingly
    trajectory, prolonged_spike_stats = detect_spike_seqs(
        trajectory, step_col, frame_col, x, y, thresh, 
        fps, seconds, spike_seq_method)
    trajectory, prolonged_spike_stats = detect_prolonged_spikes(
        trajectory, step_col, frame_col, x, y, thresh, 
        fps, seconds, spike_seq_method)
    
    # merge outlier stats
    stats = {**prolonged_spike_stats, **spike_stats} 
       
    # remove spikes which are the lower or upper bound of a sequence
    stats['spike_idxs'] = [i for i in stats[
        'spike_idxs'] if i not in stats['prolonged_spike_idxs']]
            
    # create new combined total of data points deemed to be spikes
    stats['num_detected_spikes'] = len(stats['spike_idxs'])
    stats['total_num_detected_spikes'] = prolonged_spike_stats[
          'num_detected_prolonged_spikes'] + stats[
          'num_detected_spikes'] + stats[
          'num_expected_spikes']
    
    # label outlying trajectories if total number of outliers exceeds threshold
    num_data_points = min(trajectory[[x, y]].notnull().sum())
        
    if (stats['total_num_detected_spikes'] / num_data_points) * 100 >= corrupt_thresh:
        stats['corruption_likelihood'] = 'positive'
    else:
        stats['corruption_likelihood'] = 'negative'
    
    return trajectory, stats


def detect_spikes(trajectory, step_col='stepLength', frame_col='globalFRAME',
                  segment_col='chunk_segment', x='x', y='y', thresh=5, fps=5, 
                  seconds=1, method='interpolate'):
    """
    Removes point outliers, or spikes, from a movement track by identifying
    locations with extreme incoming and outgoing step lengths that surpass a
    user-defined upper threshold. Detected spikes can in turn either kept or
    linearly interpolated.
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        segment_col (str, default='chunk_segment'): DataFrame column 
            containing count of trajectory segment.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, deafult=5): Maximum step length between frames.
        fps (int, default=5): number of frames per second.
        seconds (int, default=1): number of seconds for which to exclude
            rows before commencing outlier detection.
        method (str, default='interpolate'): method for handling spikes, 
            can be either 'interpolated' or 'keep'.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """
    # instantiate dict to collect stats
    stats = {}
    
    # append step length between frames to column
    check_step_length_col(trajectory)
    
    # detect expected spikes that occur at beginning of each recording
    idx_sec = seconds * fps
    expected_spikes = []
    for segment in trajectory[segment_col].unique():
        df = trajectory.loc[(trajectory[segment_col]==segment)]
        # detect spike if it exceeds threshold and occurs within first idx_sec 
        expected_segment_spikes = list(
            df.loc[(df[step_col] >= thresh) & 
                   (df.index <= (df.index[0] + idx_sec))][frame_col])
        # append all expected spikes and data points that occur in first idx_sex frames
        if len(expected_segment_spikes) > 0:
            [expected_spikes.append(i) for i in list(
                range(df.index[0], max(expected_segment_spikes)+1))]
    
    # get index of expected spikes
    expected_spike_idxs = trajectory.loc[trajectory[frame_col].isin(
        expected_spikes)].index
    
    # reset start frame by listwise excluding expected spikes and all preceding frames  
    trajectory = trajectory.drop(expected_spike_idxs) 
    
    # detect absoloute spikes with thresh-exceeding incoming step lengths
    steps = trajectory[[frame_col, step_col]]
    
    # detect spikes with thresh-exceeding incoming and outgoing step lengths
    steps['nextStep'] = steps.stepLength.shift(-1)
    spike_idxs = steps.loc[(steps[step_col]>=thresh) &
                           (steps['nextStep']>=thresh)][frame_col]
    spike_idxs = list(spike_idxs)
    
    # label spikes as NaN and then linearly linearly or keep
    if method == 'interpolate':
        trajectory.loc[trajectory[frame_col].isin(
            spike_idxs), [x, y]] = np.nan
        trajectory[[x, y]] = trajectory[[x, y]].interpolate()     
    elif method == 'keep':
        pass   
    
    # reset globalFRAME
    trajectory[frame_col] = pd.RangeIndex(start=0, stop=len(trajectory), step=1)
    
    # update step length 
    trajectory[step_col] = lbt_metrics.get_step_len(
        trajectory[x], trajectory[y])
    
    # return outlier handling statistics
    stats['num_expected_spikes'] = len(expected_spikes)
    stats['num_detected_spikes'] = len(spike_idxs)  
    stats['spike_idxs'] = spike_idxs
    
    return trajectory, stats


def detect_spike_seqs(trajectory, step_col='stepLength', frame_col='globalFRAME',
                      x='x', y='y', thresh=5, fps=5, seconds=1, 
                      spike_seq_method='exclude'):
    """
    Drops consecutive sequences of point outlers from a movement track or 
    excludes them (marks them as 'nan') by identifying the bounds and 
    removing positions between them. Detected spikes can in turn either
    dropped or linearly interpolated.
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, deafult=5): Maximum step length between frames.
        fps (int, default=5): number of frames per second.
        seconds (int, default=1): number of seconds for which to exclude
            rows before commencing outlier detection.
        spike_seq_method (str, default='interpolate'): method for handling
            spikes, can be either 'interpolated' or 'exclude', which labels 
            data points as NaN.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """
    # instantiate dict to collect stats
    stats = {}
    
    # append step length between frames to column
    check_step_length_col(trajectory)
    
    # detect spikes with thresh-exceeding incoming and outgoing step lengths
    spikes = trajectory.loc[trajectory[step_col] >= thresh][[frame_col, step_col]]

    # find first of next points with speed out > thresh
    spikes['nextSpikeFRAME'] = spikes.globalFRAME.shift(-1)

    # store prolonged spikes as those where nextSpikeFRAME += 1 and handle accordingly
    prolonged_spikes = []
    for spike in spikes.globalFRAME:
        if float(spikes.loc[spikes[frame_col]==spike].nextSpikeFRAME) == (spike+1):
            prolonged_spikes.append([spike])
            spike +=1

    # store prolonged spike indicies
    prolonged_spikes_idxs = list(set(pd.core.common.flatten(prolonged_spikes)))

    # handle spike sequences accordingly
    for spike in prolonged_spikes_idxs:
        # handle the exception of NaN values 
        if spikes.loc[spikes[frame_col]==spike].nextSpikeFRAME.isnull().sum() == False:
            # handle spikes
            if spike_seq_method == 'interpolate':
                # label spikes as NaN and then interpolate linearly
                trajectory.loc[trajectory[frame_col].isin(
                    prolonged_spikes_idxs), [x, y]] = np.nan
                trajectory[[x, y]] = trajectory[[x, y]].interpolate()

            elif spike_seq_method == 'exclude':
                # drop spikes
                trajectory.loc[trajectory[frame_col].isin(
                    prolonged_spikes_idxs), [x, y, step_col]] = np.nan
                
    # reset globalFRAME
    trajectory[frame_col] = pd.RangeIndex(start=0, stop=len(trajectory), step=1)
    
    # return outlier handling statistics
    stats['num_detected_prolonged_spikes'] = len(prolonged_spikes_idxs)
    stats['prolonged_spike_idxs'] = list(prolonged_spikes_idxs)
   
    return trajectory, stats


def detect_prolonged_spikes(trajectory, step_col='stepLength', frame_col='globalFRAME',
                            x='x', y='y', thresh=5, fps=5, seconds=1, 
                            spike_seq_method='exclude'):
    """
    Drops subsequence outliers, or prolonged spikes from a movement track or 
    excludes them (marks them as 'nan') by identifying the bounds and 
    removing positions between them. Detected spikes can in turn either
    dropped or linearly interpolated.
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, deafult=5): Maximum step length between frames.
        fps (int, default=5): number of frames per second.
        seconds (int, default=1): number of seconds for which to exclude
            rows before commencing outlier detection.
        spike_seq_method (str, default='interpolate'): method for handling spikes, 
            can be either 'interpolated' or 'exclude', which labels data
            points as NaN.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """
    # instantiate dict to collect stats
    stats = {}

    # append step length between frames to column
    check_step_length_col(trajectory)

    # detect spikes with thresh-exceeding incoming and outgoing step lengths
    spikes = trajectory.loc[trajectory[step_col] >= thresh][[frame_col, step_col]]

    # find first of next points with speed out > thresh
    spikes['nextSpikeFRAME'] = spikes.globalFRAME.shift(-1)

    # store initial spikes as those where nextSpikeFRAME =+ 1 and handle spike sequences accordingly
    spikes_idxs = []
    for spike in spikes.globalFRAME:
        next_spike = spikes.loc[spikes[frame_col]==spike].nextSpikeFRAME.sum()
        if next_spike > (spike+1):
            spikes_idxs.append([int(spike), int(next_spike)])
            spike +=1

    # store spike indicies
    spikes_idxs = set(pd.core.common.flatten(spikes_idxs))

    # detect prolonged spikes and handle accordingly
    prolonged_spikes_idxs = []
    spike_edges = []
    for spike in spikes_idxs:
        # handle the exception of NaN values 
        next_spike = spikes.loc[spikes[frame_col]==spike].nextSpikeFRAME.sum()
        if spike not in spike_edges and pd.isnull(next_spike) == False:
            # define prolonged spike 
            prolonged_spike_idx = list(range(int(spike), int(next_spike)))
            prolonged_spikes_idxs.append(prolonged_spike_idx)
            spike_edges.append(spike)
            spike_edges.append(next_spike)
            # handle prolonged spike
            if spike_seq_method == 'interpolate':
                # label as NaN and then interpolate linearly
                trajectory.loc[trajectory[frame_col].isin(
                    prolonged_spike_idx), [x, y]] = np.nan
                trajectory[[x, y]] = trajectory[[x, y]].interpolate()

            elif spike_seq_method == 'exclude':
                # drop spikes
                trajectory.loc[trajectory[frame_col].isin(
                    prolonged_spike_idx), [x, y, step_col]] = np.nan
        else:
            pass

    # reset globalFRAME
    trajectory[frame_col] = pd.RangeIndex(start=0, stop=len(trajectory), step=1)

    # return outlier handling statistics
    stats['num_detected_prolonged_spikes'] = len(
        set(pd.core.common.flatten(prolonged_spikes_idxs)))
    stats['prolonged_spike_idxs'] = list(prolonged_spikes_idxs)

    return trajectory, stats


def exclude_step_lengths(trajectory,step_col='stepLength', frame_col='globalFRAME',
                          x='x', y='y', thresh=5, method='exclude'):
    """
    Listwise drops false step length metrics from a movement or excludes them 
    (marks them as NaN).  
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, default=5): Maximum step length between frames.
        method (str, default='nan'): decides whether to drop or label
            thresh-exceeding spikes as 'exclude'.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """
    # instantiate dict to collect stats
    stats = {}
    
    # append step length between frames to column
    check_step_length_col(trajectory)

    # detect absoloute spikes with thresh-exceeding incoming step lengths
    steps = trajectory[[frame_col, step_col]]
    spike_idxs = steps.loc[steps[step_col]>=(thresh)][frame_col]

    # determine whether they precede an excluded data point
    excess_spike_idxs = []
    for i in spike_idxs:
        if np.isnan(float(steps[step_col].loc[steps[frame_col]==(i-1)])):
            excess_spike_idxs.append(i)

    # drop spikes based on index values or label as nan
    if method == 'exclude':
        trajectory.loc[trajectory[frame_col].isin(
            excess_spike_idxs), [step_col]] = np.nan

    elif method == 'drop':
        trajectory = trajectory.drop(list(excess_spike_idxs))

    # reset globalFRAME
    trajectory[frame_col] = pd.RangeIndex(start=0, stop=len(trajectory), step=1)

    return trajectory, stats
    

def exclude_interpolated_points(trajectory, step_col='stepLength', 
                                frame_col='globalFRAME',x='x', y='y',
                                thresh=5, method='exclude'):
    """
    Listwise excludes point outliers, or spikes, from a movement that exceed a 
    predefined tep length. 
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        frame_col (str, default='globalFRAME'): DataFrame column containing
            frame count.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        thresh (int, default=5): Maximum step length between frames.
        method (str, default='nan'): decides whether to drop or label
            thresh-exceeding spikes as 'exclude'.
        
    Returns:
        trajectory (pandas.DataFrame): movement track with outliers 
            excluded or interpolated.
        stats (dict): excluded frame count and number of outliers detected
            count.
    """
    # instantiate dict to collect stats
    stats = {}
    
    # append step length between frames to column
    check_step_length_col(trajectory)

    # detect absoloute spikes with thresh-exceeding incoming step lengths
    steps = trajectory[[frame_col, step_col]]
    excess_spike_idxs = steps.loc[steps[step_col]>=(thresh)][frame_col]

    # drop spikes based on index values or label as nan
    if method == 'exclude':
        trajectory.loc[trajectory[frame_col].isin(
            excess_spike_idxs), [x, y, step_col]] = np.nan

    elif method == 'drop':
        trajectory = trajectory.drop(list(excess_spike_idxs))

    # reset globalFRAME
    trajectory[frame_col] = pd.RangeIndex(start=0, stop=len(trajectory), step=1)

    # return outlier handling statistics
    stats['num_excess_spikes'] = len(excess_spike_idxs)  
    stats['excess_spike_idxs'] = excess_spike_idxs
    
    return trajectory, stats


def check_step_length_col(trajectory, step_col='stepLength', x='x', y='y'):
    """
    Add column to trajectory with step length between rows of x
    and y-coordinates.
    
    Args:
        trajectory (pandas.DataFrame): movement track.
        step_col (str, default=stepLength): DataFrame column containg step
            lengths.
        x (str, default='x'): DataFrame column containing x-coordinate.
        y (str, default='y'): DataFrame column containing y-coordinate.
        
    Returns:
        trajectory.
    """
    if step_col not in trajectory.columns:
        trajectory[step_col] = lbt_metrics.get_step_len(
            trajectory[x], trajectory[y])
    
    return trajectory