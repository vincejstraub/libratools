#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes NPZ files generated from Loopbio and CSV files generated from 
BioTracker using libratools.
"""

import os    # standard library
import warnings
import pathlib
import argparse
import configparser

import yaml    # 3rd party packages
import pandas as pd

from libratools import lbt_utils   # local imports
from libratools import lbt_impute
from libratools import lbt_metrics
from libratools import lbt_inspect
from libratools import lbt_datasets
from libratools import lbt_outlier_detection

__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"

warnings.filterwarnings("ignore")


CONFIG_PATH = pathlib.Path.cwd() / 'config.ini'
DELIM_CHAR = '\\'


def main(DATE):
    """
    Calls date input and runs processing pipeline function.
    """
    # get params needed to locate and load data
    cameras, date, datetime_date = get_input_args(DATE)
    
    try:
        camera_ids = [cameras[cam]['ID'] for cam, vals in cameras.items()]      
        # print informational message
        print(f'Processing data for {datetime_date} for cameras: {camera_ids}.\n') 
        # run processing pipeline
        for cam in cameras.keys():
            print(f'Processing camera: {cameras[cam]["ID"]}.')
            run_pipeline(cameras, cam, date) 
    except AttributeError:
        raise AttributeError('Check camera(s) in camera_ids.yaml are not commented out.') 


def get_input_args(DATE):
    """
    Returns camera ids and date for which to process CSV files.
    """
    # set date argument to yesterday if none provided
    if not DATE:
        date = lbt_utils.get_date(delta=-1)
    else:
        # validate provided date argument
        date = DATE
        lbt_utils.validate_date_arg(date)

    # change to datetime format for printing
    datetime_date = lbt_utils.strptime_date_arg(date)

    # get list of camera labels and serial numbers
    camera_ids_path = pathlib.Path(
        HOME_DIR, REPO_ROOT_DIR, REPO_PROCESSIG_DIR, CAMERAS_IDS)
    with open(camera_ids_path, 'r') as stream:
        try:
            cameras = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cameras, date, datetime_date


def run_pipeline(cameras, cam, date):
    """
    Run processing pipeline, saving processed trajectory to disk.
    """
    # get file paths
    file_paths, biotracker_metadata_file, segments_dir = locate_data(
        cameras, cam, date)
    # load data optionally saving merged trajectory to disk
    trajectory_raw, file_name = load_data(file_paths, biotracker_metadata_file,
                                          cameras,
                                          cam,
                                          save_merged=True,
                                          outdir=segments_dir)
    
    # handle missing data
    trajectory_merged, missing_vals, total_tracking_time, row_count = preprocess_data(
        trajectory_raw)

    # compute movement metrics and detect outliers
    trajectory_processed = postprocess_data(cameras, cam, trajectory_merged, 
                                            missing_vals, total_tracking_time,
                                            row_count)

    # store processed metadata as comments
    processing_metadata = lbt_datasets.dict_to_comments(
        trajectory_processed['metadata'])

    # save processed trajectory along with metadata
    lbt_datasets.save_trajectory_to_csv(
        trajectory_processed['data'], metadata=processing_metadata,
        f_name=file_name, outdir=segments_dir, save_msg=True, add_metadata=True)     
    
    
def locate_data(cameras, cam, date):
    """
    Returns file paths to loopbio NPZ files and Biotracker-generated CSV files.
    """
    #declare data path
    data_path = pathlib.Path(HOME_DIR, DATA_ROOT_DIR)

    # declare directory path based on date argument   
    date_dir = lbt_datasets.find_dir(
        path=data_path,
        prefix=date,
        suffix=cameras[cam]['ID'])
    
    # declare trajectory segments path based on date argument and camera
    try:
        segments_dir = data_path / str(cameras[cam]['ID']) / date_dir
    except TypeError:
        print('TypeError: unsupported operand type(s) for /: "WindowsPath" and "NoneType".')
        print('Check DATA_DIR in config.ini is correct and contains no backslashes.')
        
    # load NPZ chunk and CSV segment file paths for processing
    loopbio_file_paths, num_loopbio_chunks = lbt_datasets.read_file_paths(
        indir=segments_dir,
        extension=LOOPBIO_ARRAY_EXTENSION,
        warning=False)
    
    biotracker_file_paths, num_biotracker_segments = lbt_datasets.read_file_paths(
        indir=segments_dir,
        extension=BIOTRACKER_FILE_EXTENSION,
        warning=False)

    # assert number of csv and npz chunks is consistent with loopbio settings
    if num_loopbio_chunks != CHUNK_COUNT:
        msg = f'{num_loopbio_chunks} {LOOPBIO_ARRAY_EXTENSION} files in \
                {segments_dir}'
        warning = ', expected: 'f'{CHUNK_COUNT}.'
        print(msg + warning)
    if num_biotracker_segments != CHUNK_COUNT:
        msg = f'{num_biotracker_segments} {BIOTRACKER_FILE_EXTENSION} \
                 files in {segments_dir}'
        warning = ', expected: 'f'{CHUNK_COUNT}.'
        print(msg + warning)

    # combine array and segment file paths in a dictionary for iteration
    file_paths = dict(zip(loopbio_file_paths, biotracker_file_paths))

    return file_paths, biotracker_file_paths, segments_dir


def load_data(file_paths, biotracker_metadata_file, cameras, cam,
              save_merged=True, outdir=''):
    """
    Returns merged trajectory segments as pandas.DataFrame.
    """
    # iterate over segment file paths, storing segments that are not corrupt
    trajectory_dfs = {}
    for loopbio_chunk_path, biotracker_segment_path in file_paths.items():
        # get video chunk number
        chunk_num = lbt_datasets.get_chunk_number_from_path(loopbio_chunk_path,
                                                            dir_sep=DELIM_CHAR)

        # load Loopbio NPZ frame numbers into numpy array
        loopbio_arr = lbt_datasets.load_npz(loopbio_chunk_path)
        loopbio_frames = loopbio_arr['frame_number']

        # get number of frames dropped by loopbio
        dropped_frames = lbt_utils.count_dropped_frames(loopbio_frames)

        # load BioTracker segment into DataFrame
        segment_df, _ = lbt_datasets.load_trajectory(biotracker_segment_path)

        # add frame times as new datetime column rounded to 3 decimal places
        segment_df['timestamp'] = pd.Series(
            [lbt_utils.unixtime_to_strtime(ts)[:-3] for ts in
             loopbio_arr['frame_time']])
        segment_df['timestamp'] = pd.to_datetime(segment_df['timestamp'])
        # update timezone to host timezone
        segment_df['timestamp'] = pd.Series(
            [lbt_utils.to_local_datetime(ts) for ts in
             list(segment_df['timestamp'])])

        # decide whether to keep or drop segment based on nans in key columns
        segment_dfs, num_dfs = lbt_utils.partition_segment(
            segment_df, segment_df.MillisecsByFPS, chunk_num, cols=['x', 'y'])

        # append segment DataFrame to dict if it has not been corrupted
        if num_dfs > 0:
            for segment in range(0, num_dfs):
                # append along with experimental setup and loopbio metadata
                key = f'{str(chunk_num)}' + '_' + f'{segment}'
                trajectory_dfs[key] = {'data': segment_dfs[segment],
                                       'metadata': {
                                                'dropped_frames': dropped_frames
                                             }
                                              }    
                
    # store BioTracker-generated metadata from first chunk
    biotracker_metadata = lbt_datasets.read_metadata(
        biotracker_metadata_file[0])
    
    # replace BioTracker-stored FPS with FPS set in config.ini
    biotracker_metadata[1] = f'# {FPS}\n'
                
    # combine BioTracker-generated metadata with trajectory overview metadata
    overview_metadata = dict({'camera_id': cameras[cam]['ID'],
                              'camera_num': cameras[cam]['number'],
                              'color': cameras[cam]['color'],
                              'experimental_group': cameras[cam]['group']})
                               
    # append overview metadata to track dict
    overview_metadata = [f'# {key}: {value}\n' for key, value in overview_metadata.items()]
    merged_metadata = biotracker_metadata + overview_metadata

    # merge chunks as single trajectory, saving merged raw file to disk
    trajectory = lbt_utils.aggregate_segments(
        trajectory_dfs, metadata=merged_metadata,
        save_trajectory=True,
        file_name=biotracker_segment_path,
        outdir=outdir,
        suffix='_merged', save_msg=True)
    
    # sort DataFrame
    trajectory['data'].sort_values(by=['globalFRAME'], inplace=True)

    return trajectory, biotracker_segment_path


def preprocess_data(trajectory):
    """
    Load and impute missing merged trajectory.
    """
    # drop missing values at beginning and end, clean up other columns
    trajectory['data'] = lbt_impute.fill_missing(trajectory['data'])

    # count missing values and add to metadata
    missing_vals = lbt_utils.count_missing_values(
        trajectory['data'], ['x', 'y'])
    
    # count rows with missing values
    incomplete_rows = lbt_utils.sum_dict_vals(
        missing_vals, keys=['x_nans', 'y_nans'])

    # get total row count for which data points exist
    row_count = int(len(trajectory['data']) - incomplete_rows)
    
    # get total tracking time elapsed for which data points exist in minutes
    time_between_frames = 1 / FPS
    total_tracking_time = lbt_inspect.time_change_from_frame_count(
        trajectory['data']['globalFRAME'].iloc[-1], 
        time_between_frames)

    # handle missing coordinate values depending on default method and thresh
    missing_vals_per = (incomplete_rows / len(trajectory['data'])) * 100

    if missing_vals_per < MAX_MISSING_VAL_PER:

        if MISSING_VAL_METHOD == 'interpolate':
            # columns for which to interpolate values
            fill_cols = ['x', 'y', 'rad', 'deg', 'xpx', 'ypx']
            trajectory['data'] = lbt_impute.interpolate_nan_vals(
                trajectory['data'], cols=fill_cols)

        elif MISSING_VAL_METHOD == 'drop':
            trajectory['data'] = lbt_utils.drop_missing_coord_vals(
                trajectory['data'])

        elif MISSING_VAL_METHOD == 'pass':
            pass
    else:
        pass

    return trajectory, missing_vals, total_tracking_time, row_count


def postprocess_data(cameras, cam, trajectory, 
                     missing_vals, total_tracking_time, row_count):
    """
    Compute metrics and detect outlying trajectories.
    """
    # sort DataFrame
    trajectory['data'].sort_values(by=['globalFRAME'], inplace=True)
    
    # get step length (element-wise euclidean distance) between frames to detect outliers
    trajectory['data']['stepLength'] = lbt_metrics.get_step_len(
        trajectory['data']['x'], trajectory['data']['y'])

    # run outlier (spike) detection methods
    trajectory['data'], spike_stats = lbt_outlier_detection.run_detection(
        trajectory['data'], step_col='stepLength', frame_col='globalFRAME', x='x', y='y', 
        thresh=MAX_STEP_THRESH, fps=FPS, seconds=SKIP_INITIAL, spike_method=OUTLIER_METHOD,
        spike_seq_method=GROUPED_OUTLIER_METHOD, corrupt_thresh=MAX_OUTLIERS_VAL_PER)
            
    # update step length and append to trajectory
    trajectory['data']['stepLength'] = lbt_metrics.get_step_len(
        trajectory['data']['x'], trajectory['data']['y'])
    
    # exclude any excess step lengths exceeding the jump threshold 
    trajectory['data'], excess_vals = lbt_outlier_detection.exclude_step_lengths(
        trajectory['data'], thresh=MAX_STEP_THRESH, method='exclude')

    # exclude any interpolated data points with step lenghts exceeding the threshold
    trajectory['data'], excess_vals = lbt_outlier_detection.exclude_interpolated_points(
        trajectory['data'], thresh=MAX_STEP_THRESH, method='exclude')
            
    # update step length and append to trajectory once more
    trajectory['data']['stepLength'] = lbt_metrics.get_step_len(
        trajectory['data']['x'], trajectory['data']['y'])

    # get total count of rows excluded or dropped
    rows_removed_count = int(
        (spike_stats['num_expected_spikes'] + excess_vals['num_excess_spikes']) + (
         trajectory['metadata']['dropped_frames']))
        
    # append relative turning angle between frames to column
    trajectory['data']['turnAngle'] = lbt_metrics.get_relative_turn_angle(
        trajectory['data'])
    
    # run diagnostics to get summary statistics
    summary_stats = lbt_inspect.summarize(
        trajectory['data'], trajectory['data']['x'], trajectory['data']['y'])
        
    # append experimental setup and diagnostics metadata to track dict
    preprocess_metadata = dict({'camera_id': cameras[cam]['ID'],
                                'camera_num': cameras[cam]['number'],
                                'color': cameras[cam]['color'],
                                'experimental_group': cameras[cam]['group'],
                                'total_tracking_time_mins': total_tracking_time,
                                'row_count': row_count,
                                'rows_removed_count': rows_removed_count},
                                **missing_vals,
                                **spike_stats,
                                **excess_vals,
                                **summary_stats
                              )
    
    # append diagnostics metadata to track dict
    for k, v in preprocess_metadata.items():
        trajectory['metadata'][k] = v
        
    # sort DataFrame
    trajectory['data'].sort_values(by=['globalFRAME'], inplace=True)

    return trajectory


class configReader:
    __conf = None

    @staticmethod
    def config():
        if configReader.__conf is None:  # read only once, lazy
            configReader.__conf = configparser.ConfigParser()
            configReader.__conf.read(CONFIG_PATH)
        return configReader.__conf


if __name__ == '__main__':
    # add optional date argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', nargs='?', default=None,
                        help="Date for which to process data in the format: \
                              YYYYMMDD. If not is provided yesterday's date \
                              is used by default.")
    args = parser.parse_args()
    # read directory configuration for global vars
    REPO_ROOT_DIR = configReader.config()['PATHS']['REPO_ROOT_DIR']
    DATA_ROOT_DIR = configReader.config()['PATHS']['DATA_DIR']
    REPO_PROCESSIG_DIR = configReader.config()['PATHS']['REPO_PROCESSING_DIR']
    LOOPBIO_METADATA = configReader.config()['FILES']['LOOPBIO_METADATA']
    LOOPBIO_ARRAY_EXTENSION = configReader.config()['LOOPBIO']['ARRAY_EXTENSION']
    LOOPBIO_ARRAY_EXTENSION = configReader.config()['LOOPBIO']['ARRAY_EXTENSION']
    BIOTRACKER_FILE_EXTENSION = configReader.config()['BIOTRACKER']['FILE_EXTENSION']
    CAMERAS_IDS = configReader.config()['FILES']['CAMERAS_IDS']
    # read default settings
    FPS = int(configReader.config()['LOOPBIO']['FPS'])
    CHUNK_COUNT = float(
        configReader.config()['LOOPBIO']['CHUNK_COUNT'])
    SKIP_INITIAL = float(
        configReader.config()['DEFAULTS']['SKIP_INITIAL'])
    MAX_MISSING_VAL_PER = float(
        configReader.config()['DEFAULTS']['MAX_MISSING_VAL_PER'])
    MISSING_VAL_METHOD = configReader.config()['DEFAULTS']['MISSING_VAL_METHOD']
    OUTLIER_METHOD = configReader.config()['DEFAULTS']['OUTLIER_METHOD']
    GROUPED_OUTLIER_METHOD = configReader.config()['DEFAULTS']['GROUPED_OUTLIER_METHOD']
    MAX_OUTLIERS_VAL_PER = float(
        configReader.config()['DEFAULTS']['MAX_OUTLIERS_VAL_PER'])
    MAX_STEP_THRESH = float(
        configReader.config()['DEFAULTS']['MAX_OUTLIERS_VAL_PER'])
    # store home diretory where repo is saved
    HOME_DIR = pathlib.Path(os.getcwd()).parents[1]
    # run script
    if args.date is None:
        main(None)
    else:
        lbt_utils.validate_date_arg(args.date)
        main(args.date)
