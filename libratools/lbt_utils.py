#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_utils module includes various utilities and private
functions.
"""

import datetime     # standard library        
import calendar

import statistics    # 3rd party packages
import numpy as np
import pandas as pd

from . import lbt_datasets   # local imports


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


def main():
    pass


def count_dropped_frames(x):
    """
    Returns number of NaN frames in provided array along.
    """
    x_arr = np.array(x)
    num_nans = np.count_nonzero(np.isnan(x_arr))

    return num_nans


def count_missing_values(df, cols, idxs=False):
    """
    Returns the number of rows with missing values for each provided
    column of a DataFrame as well as the total number for all columns.

    Args:
        df (DataFrame): pandas DataFrame.
        cols (list): pandas DataFrame columns to check for missing row values.
        idxs (bool, default=False): if idxs=True, the index for each row is
            also returned.

    Returns:
        A dict of column name as the key and a tuple of the number of
            misssing rows and the index for each row as values.
    """
    na_rows = {}

    # collect the total number of missing rows and row index
    missing_val_idxs = list(df.index[df.isna().any(axis=1)])
    if len(missing_val_idxs) > 0:
        na_rows['total_nans_across_cols'] = df.isna().values.sum()
    else:
        na_rows['total_nans_across_cols'] = 0

    # collect the number of missing rows and row index for each column
    for col in list(cols):
        col_name = col + '_nans'
        missing_val_idxs = df[df[col].isnull()].index.tolist()
        if len(missing_val_idxs) > 0:
            na_rows[col_name] = len(missing_val_idxs)
            if idxs is True:
                na_rows[col_name + '_idxs'] = missing_val_idxs
        else:
            na_rows[col_name] = 0

    return na_rows


def partition_segment(df, time_series, segment_num, cols=['x', 'y'], thresh=5,
                      unit='seconds', segment_limit=2, data_loss_limit=0.5):
    """
    Returns or discards dataset using the following decision rule:
    if more than 50% (default value) of recording is in tact in no more
    than 2 (default value) continuous segments the dataset is deemed worth using
    and is kept. Continuous is defined by default as 'no time interval gap
    between consecutive frames greater than 5 seconds (5000 milliseconds)'.
    If the dataset is split, a list of dataframes is returned, if discarded,
    None is returned.

    Args:
        df (DataFrame): pandas DataFrame containing time stamps.
        time_series (Series): list of values containing time stamps/frame value.
        cols (list): pandas DataFrame columns to check for missing row values.
        missing_vals (list): number of missing rows for key columns.
        thresh (int, default=5): threshold for deciding when to create
            new DataFrame based on change in time interval value.
        unit (str, default=seconds): time unit for thresh value.
        segment_limit (int, default=4): maximum number of continuous blocks.
        data_loss_limit (float, default=0.5): threshold that determines maximum 
            amount of data loss permitted before a dataset is discarded.
    Returns:
        DataFrames and number of DataFrames as integer.
    """
    if unit == 'seconds':
        thresh_val = thresh * 1000
    elif unit == 'milliseconds':
        pass
    else:
        ValueError("Time unit must be in seconds or milliseconds.")

    # compute number of missing vals then drop these rows from a dummy df
    nan_percent = (df[cols].isnull().sum() / len(df)).sum()
    temp_df = df.copy()
    temp_df.dropna(inplace=True)

    # compute time change between frames and create new column
    temp_df['MillisecsBetweenFRAMES'] = (time_series - time_series.shift())

    # append a new DataFrame when time interval value exceeds threshold
    temp_dfs = {}
    for _, g in temp_df.groupby((temp_df.MillisecsBetweenFRAMES.diff() 
                                 > thresh_val).cumsum()):
        temp_dfs[_] = g

    dfs = {}
    # implement decision rule to decide whether to keep dataset
    if len(temp_dfs.keys()) > segment_limit and nan_percent <= data_loss_limit:
        print(f'Skipped segment {segment_num}, segment limit surpassed.')
        return dfs, 0
    elif len(temp_dfs.keys()) <= segment_limit and nan_percent > data_loss_limit:
        print(f'Skipped segment {segment_num}, data loss limit surpassed.')
        return dfs, 0
    elif len(temp_dfs.keys()) > segment_limit and nan_percent > data_loss_limit:
        print(f'Skipped segment {segment_num}, data loss and segment limit surpassed.')
        return dfs, 0
    else:
        df['MillisecsBetweenFRAMES'] = (time_series-time_series.shift())
        # return original DataFrame with nans
        for _, g in df.groupby((df.MillisecsBetweenFRAMES.diff() > thresh_val).cumsum()):
            dfs[_] = g
        # return dfs and number of dfs for indexing
        num_dfs = len(dfs.keys())

        return dfs, num_dfs


def aggregate_segments(dfs, time_interval=40, save_trajectory=False,
                       metadata=[''], file_name='', suffix='_processed',
                       save_msg=True, outdir=''):
    """
    Returns DataFrame of aggregated segments, individual DataFrames, optionally
    saving combined file to disk using index as reference column to maintain
    global FRAME count.

    Args:
        dataframes (dict): DataFrames of each segment.
        time_interval (int, default=40): modal time interval.
        save_trajectory (bool, default=True): if save_trajectory=True, aggregated
            CSV file is saved to disk.
        metadata (list, default=['']): list of strings to prepend to CSV file
            if saved to disk.
        outdir (str, default='.'): path of directory where to save aggregated
            file, default is current working directory.
        file_name (str, default=''): file name.
        suffix (str, default='_processed'): suffix to add to the end of file
            when saving.
    Returns:
        Dict of aggregated DataFrame and list of metadata
    """
    all_dfs = {}
    dropped_frames_counter = 0
    MillisecsByFPS_counter = 0
    # load CSV and append to list with segment file name as new column
    for segment in dfs.keys():
        # create new local FRAME column
        dfs[segment]['data'].insert(
            loc=1, column='localFRAME', value=dfs[segment]['data'].index)
        # set FRAME as index to maintain cumulative count when concatenating
        dfs[segment]['data'].set_index('FRAME', inplace=True)
        # create new column which stores video chunk segment number
        dfs[segment]['data']['chunk_segment'] = segment
        # add last MillisecsByFPS value to counter to maintain cumulative count
        if segment == list(dfs.keys())[0]:
            last_MillisecsByFPS_val = dfs[segment]['data']['MillisecsByFPS'].iloc[-1]
            # add time_interval as MillisecsByFPS starts at 0l
            MillisecsByFPS_counter += (last_MillisecsByFPS_val + time_interval)
            # add 0 to first row of first segment
            dfs[segment]['data'].at[0, 'MillisecsBetweenFRAMES'] = 0
        else:
            last_MillisecsByFPS_val = dfs[segment]['data']['MillisecsByFPS'].iloc[-1]
            # skip first segment when adding MillisecsByFPS_counter
            dfs[segment]['data']['MillisecsByFPS'] = dfs[segment]['data']['MillisecsByFPS'] + \
                MillisecsByFPS_counter
            MillisecsByFPS_counter += (last_MillisecsByFPS_val + time_interval)
            # add time_interval to first row of subsequent segments
            dfs[segment]['data'].at[0, 'MillisecsBetweenFRAMES'] = time_interval
        all_dfs[segment] = {'data': dfs[segment]['data']}
        dropped_frames_counter += dfs[segment]['metadata']['dropped_frames']

    # aggregate DataFrames and insert global FRAME as a separate column
    list_of_dfs = [all_dfs[i]['data'] for i in all_dfs.keys()]
    aggregate_df = pd.concat(list_of_dfs, ignore_index=True)
    aggregate_df.insert(loc=0, column='FRAME', value=aggregate_df.index)
    aggregate_df.rename(columns={'FRAME': 'globalFRAME'}, inplace=True)

    # clean metadata by removing comment chars
    metadata_clean = [i.split('#')[1].strip() for i in metadata]

    # add dropped frames, missing values count and metdata to dict
    aggregate_data_metadata_df = {
        'data': aggregate_df,
        'metadata':{
             'source_name': metadata_clean[0],
             'source_fps': metadata_clean[1],
             'generation_time': metadata_clean[2],
             'dropped_frames': dropped_frames_counter},
    }

    # save aggregated DataFrame to disk as CSV and/or return for preprocessing
    if save_trajectory is True:
        # drop video chunk number from file name
        lbt_datasets.save_trajectory_to_csv(
            aggregate_df, metadata=metadata, f_name=file_name,
            outdir=outdir, suffix=suffix, save_msg=save_msg)

    return aggregate_data_metadata_df


def split_series_on_datestr(df, col, HH='', MM='', SS='', YYYYMMDD=''):
    """
    Split pandas.DataFrame into two on the basis of value in a column with
    datetime values using 'YYYY-MM-DD HH:MM:SS'
    
    Args:
        df (pandas.DataFrame): dataframe to split.
        col (pandas.Series): column to split on.
        HH (str, default=''): hour value in 24-hour clock format. 
        MM (str, default=''): minute value.
        SS (str, default=''): second value.
        
    Returns:
        df_split (pandas.DataFrame)
    """
    yyyy = YYYYMMDD[:4]
    mm = YYYYMMDD[4:6]
    dd = YYYYMMDD[-2:]
    if YYYYMMDD == '':
        yyyymmdd = str(df[col][0].to_period('D'))

    ts = f'{yyyymmdd} {HH}:{MM}:{SS}'
    split_ts = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    
    df_split = df.loc[df[col] > split_ts]
    
    return df_split


def strptime_date_arg(date):
    """
    Returns string as foramtted datetime object using the regex format:
    %Y-%M-%d (%a) where string is expected to be in the format YYYYMMDD.
    """
    try:
        int(date)
    except ValueError:
        print(f'{date} does not exclusively contain integers.')
    try:
        _ = datetime.datetime.strptime(date, '%Y%M%d')
        datetime_obj = _.strftime('%Y-%M-%d')
        return datetime_obj
    except ValueError:
        print(f'{date} is not in the format YYYYMMDD.')


def get_date(string=True, delta=0, date_format='%Y%m%d'):
    """
    Returns date for which to preprocess CSV files generated by BioTracker, the
    default behavior is to return today's date as a 'YYYYMMDD' string.

    Args:
        string (bool, default=True): if string=True, datetime object is
            returned as a string.
        delta (int, default=0): number of days to add or subtract to current
            date to get desired date, if default=0 current date is returned.
        date_format (str, default='%Y%m%d'): format for datetime string, if
            default='%Y%m%d' date will be in the form 'YYYMMMDD'.

    Returns:
        date (datetime or str).
    """
    date = datetime.datetime.now() + datetime.timedelta(delta)
    if string:
        return date.strftime(f'{date_format}')
    else:
        return date


def unixtime_to_strtime(ut, str_format='%Y-%m-%d %H:%M:%S.%f'):
    """
    Returns unix time object in datetime formatted string.

    Args:
        ut (float): unix time stamp.
        str_format(str, default='%Y-%m-%d %H:%M:%S.%f'): format
            for datetime object.

    Returns:
        ts_formatted (datetime.datetime)
    """
    ts = float(ut)
    try:
        ts = datetime.datetime.utcfromtimestamp(ts)
        ts = ts.strftime(f'{str_format}')
        return ts
    except ValueError:
        print('Unix time stamp must be in seconds, not milliseconds.')
        

def to_local_datetime(utc_dt):
    """
    Converts from utc datetime to a locally aware datetime according
    to the host timezone.
    
    Args:
        utc_dt (datetime.datetime): utc datetime object.
    
    Returns:
        local timezone datetime
    """
    return datetime.datetime.fromtimestamp(calendar.timegm(utc_dt.timetuple()))


def convert_seconds(s):
    """
    Converts seconds to minutes and hours.
    
    Args:
        s (float): value in seconds.
        
    Returns:
        mins (float): s in minutes
        hrs (float): s in hours.   
    """
    mins = s / 60
    hrs = s / 3600
    
    return mins, hrs


def convert_milliseconds(ms):
    """
    Converts milliseconds to seconds, minutes, hours, and days.

    Args:
        ms (float): value in milliseconds.

    Returns:
        MillisecsBySecsMinsHrs (list): ms in seconds, minutes, hours, days.
    """
    secs = ms / 1000.0
    mins = secs / 60.0
    hrs = mins / 60.
    MillisecsBySecsMinsHrsDays = {
        'secs': secs, 'mins': mins, 'hrs': hrs, 'days': days
    }

    return MillisecsBySecsMinsHrs


def add_nested_vals_to_dict(dic1, dic2, key3='metadata', 
                            keys2=['treatment', 'begin_treatment']):
    """
    Add values from one nested dictionary to another.
    """
    for key1 in dic1.keys():
        for key2 in keys2:
            dic1[key1][key3][key2] = dic2[key1][key2]

    return dic1


def sum_dict_vals(dic, keys=['x_nans', 'y_nans', 'globalFRAME']):
    """
    Returns maximum value of values for selective keys of a dictionary.

    Args:
        dic (dict): dict to iterate through.
        keys (list, default=['x_nans', 'y_nans',]): keys to iteratve 
            over.

    Returns:
        val (int): maximum value
    """
    val = max([dic[key] for key in dic.keys() if key in keys])

    return val


def validate_date_arg(date, date_format='YYYYMMDD'):
    """
    Validates date string is in format YYYYMMDD, prints error message
    if not.

    Args:
        date (str): date to validate.
    """
    try:
        datetime.datetime.strptime(date, '%Y%m%d')
    except ValueError:
        raise ValueError(f'Incorrect date argument, expected: {date_format}')


def strlist_to_intlist(strlist):
    """
    Returns list of strings as list of ints.
    """
    try:
        intlist = [int(i) for i in strlist]
        return intlist
    except ValueError:
        print('The list argument does not exclusively contain numbers.')


def update_setup(dic, new_vals):
    """
    Appends dictionary values to existing dictionary.
    """
    for k, v in new_vals.items():
        dic[k].append(v)

    return dic


def _is_type(obj, object_type):
    """
    Checks object type against expect type.
    """
    if not isinstance(obj, object_type):
        print(f'Input is {type(obj)}, expected {object_type}.')


def _has_cols(df, cols={'x', 'y'}):
    """
    Checks if specified columns are in pandas.DataFrame.
    """
    if not cols.issubset(df.columns):
        print(f'The following columns are all required: {cols}')


def drop_missing_coord_vals(df, cols=['x', 'y', 'rad',
                                      'deg', 'xpx', 'ypx']):
    """
    Returns DataFrame after dropping NaN values in select columns with
    index reset.
    """
    df.dropna(subset=cols, how='all', inplace=True)

    return df


def get_nested_dict_values(nested_dic, key1='metadata', key2='activity'):
    """
    Returns list of values as floats for select key of a nested dictionary.
    """
    vals = [float(nested_dic[key][key1][key2]) for key in nested_dic.keys()]

    return vals


def get_modal_col_vals(df, cols=['objectName',
                                 'valid', 'id', 'coordinateUnit']):
    """
    Returns most common values for select DataFrame columns as dict.
    """
    cols_vals = {col: statistics.mode(df[col]) for col in cols}

    return cols_vals


def zip_lists(list_a, list_b, list_c,
              key1='treatment', key2='begin_treatment'):
    """
    Appends multiple lists into a nested dictionary.
    """
    res = {a:{key1: b, key2: c} for a, b, c in zip(list_a, list_b, list_c)}
    
    return res


def get_vals_above_thresh(df, col, thresh):
    """
    Returns number of rows of a DataFrame where values in selective 
    column exceed thresh as a percentage
    """
    return df[df[col] >= thresh]


if __name__ == '__main__':
    main()
