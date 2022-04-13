#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_datasets module includes utilities to load, manipulate 
and save trajectory datasets.
"""

import os     # standard library
import yaml
import pathlib
import configparser

import locale    # 3rd party packages
import numpy as np
import pandas as pd


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


# set config file depending on whether process.py or app.py is being run
if pathlib.Path.cwd().name is 'Processing':
    CONFIG_PATH = pathlib.Path.cwd() / './libratools/libratools/config.ini'
elif pathlib.Path.cwd().name is 'DevExDashboard':
    CONFIG_PATH = pathlib.Path.cwd().parents[1] / './libratools/libratools/config.ini'


class configReader:
    __conf = None

    @staticmethod
    def config():
        if configReader.__conf is None:  # read only once, lazy
            configReader.__conf = configparser.ConfigParser()
            configReader.__conf.read(CONFIG_PATH)
        return configReader.__conf


# read directory configuration for global vars
BIOTRACKER_COLS = configReader.config()['VARS']['BIOTRACKER_COLS'].split(',\n')


def check_columns(df, columns=BIOTRACKER_COLS):
    """
    Checks there are expected number of columns, returns dataframe
    with columns redefined and index reset if not.

    Args:
        df (pandas.DataFrame): dataframe to check.
        col_num (int, default=14): used as reference for number of
            columns that should be in dataframe.
        columns (list, default=BIOTRACKER_COLS): list of reference columns
            that should be in dataframe.
    """
    if list(df.columns) != columns:
        df = df.reset_index()
        df.columns = columns
    else:
        pass
    return df


def load_trajectory(file_path, dropna=False, na_summary=True, skiprows=3,
                    warn_bad_lines=True, sep=';', cols=BIOTRACKER_COLS,
                    keycols=['FRAME', 'x', 'y']):
    """
    Loads a CSV file generated from BioTracker using Pandas and numpy.
    Note that the first 3 rows containing metadata and lines with too
    many commas are automatically dropped. Whether to drop rows with
    missing values (NaN) is left up to the user.

    Args:
        file_path (string): path to file.
        drop_na (bool, default=False): if drop_na=False NaN rows are kept, if
            drop_na=True NaN rows are dropped (where at least one element is
            missing)
        na_summary (bool, default=True): if na_summary=True, the number of
            rows dropped is displayed as an int.
        skiprows (int): number of rows to skip.
        warn_bad_lines (bool, default=True): If error_bad_lines is False,
            and warn_bad_lines is True, a warning for each “bad line” will
            be output.
        sep (str, default=';'): seperator to use.
        cols (list, default=BIOTRACKER_COLS): list of expected column values.
        keycols (list, default=['FRAME', 'x', 'y']): list of key columns to 
            check for missing values.

    Returns:
        A pandas.DataFrame.
        A numpy array.
    """
    # message to display if NaN values detected
    NA_MSG = 'Missing FRAME, x, and y values detected for file:\n'

    # read csv file
    df = pd.read_csv(file_path, skiprows=skiprows, delimiter=sep,
                     error_bad_lines=False, warn_bad_lines=warn_bad_lines)

    # check columns exist
    df = check_columns(df, columns=cols)

    # check for missing values in key columns
    if df[keycols].isna().sum().any() is True:
        # store info on rows with missing values
        num_na_rows = np.count_nonzero(df[keycols].isna())
        # decide whether to drop rows and display summary info
        if dropna and na_summary is True:
            print(NA_MSG + file_path)
            df = df.dropna()
            print('Rows dropped: {}.\n'.format(num_na_rows))
        elif dropna is False and na_summary is True:
            print(NA_MSG + file_path)
            print('Rows with missing values: {}.\n'.format(num_na_rows))
        elif dropna is True and na_summary is False:
            df = df.dropna()

        else:
            pass
    else:
        pass

    # convert timeString column to datetime
    date_time_format = '%a %b %d %H:%M:%S %Y'
    # set locale to German time for converting timeString column
    locale.setlocale(locale.LC_ALL, ('de', 'utf-8'))
    df['timeString'] = pd.to_datetime(df['timeString'], errors='ignore',
                                      format=date_time_format)
    # convert DataFrame to numpy array
    data = df.to_numpy()

    return df, data


def load_npz(file_path, array=''):
    """
    Loads NPZ file from disk and returns as numpy array, optionally
    returning a single array.

    Args:
        file_path (str): path to file.
        array (str, default=''): array key to index NPZ file, if
            array='' NPZ object is returned.

    Returns:
        f (numpy array).
    """
    f = np.load(file_path)
    if array != '':
        try:
            data = f[array]
            return data
        except KeyError:
            print(f'{array} is not a key in the NPZ file.')
    else:
        return f


def read_file_paths(indir='cwd', extension='.csv', warning=True, suffix=False,
                    suffix_str=''):
    """
    Stores paths of files with specified file type in a list; first looks 
    in current directory before asking user to provide alternative directory
    path if none are found.

    Args:
        indir (str, default='cwd'): input directory containing files.
        extension (str, default='csv'): file extension.
        warning (bool default=True): if warning=True, an informational 
            message is displayed in case no files are found.
        suffix (bool default=False): if suffix=True, only file paths 
            ending in suffix_str are returned.
        suffix_str (str, ''): suffix.

    Returns:
        A list of file paths.
    """
    # search for files in current directory if no indir is provided
    extension = extension.lower()
    extension_cap = extension.capitalize()
    if indir == 'cwd':
        file_paths = [p for p in indir.rglob(f'*{extension}')]
        num_files = len(file_paths)

        # prompt user for directory path if no none found else store paths
        if num_files == 0:
            print(f'No {extension_cap} files found in current working directory')

    # check provided directory path exists and read files
    else:
        assert os.path.exists(indir), 'Directory path not found.'
        file_paths = [p for p in indir.rglob(f'*{extension}')]

        # check CSV files exist
        num_files = len(file_paths)
        if num_files == 0 and warning is True:
            print(f'No {extension_cap} files found in {indir}.')

        if suffix is True:
            files_dir = pathlib.Path(file_paths[0]).parents[0]
            file_stems = [pathlib.Path(p).stem for p in file_paths]
            files = [f + extension for f in file_stems if
                    os.path.splitext(f)[1][-len(suffix_str):] == suffix_str]
            file_paths = [files_dir / f for f in files]

    return file_paths, num_files


def find_dir(path='cwd', prefix='', suffix=''):
    """
    Returns subdirectory path that begins and ends with specific strings by
    using os.walk() to search through all directory and file paths in root
    directory of current working directory.
    Args:
        path (str, default=cwd): directory to search, if default=cwd,
            the current working directory is searched.
        prefix (str, default=''): prefix of subdirectory path.
        suffix (str, default=''): suffix of subdirectory path.
    Returns:
        dir_path (str).
    """
    prefix_str = str(prefix)
    suffix_str = str(suffix)
    date_dir = None
    if path == 'cwd':
        path = pathlib.Path.cwd()
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith(f"{prefix_str}") and dir.endswith(f"{suffix_str}"):
                date_dir = dir

    if date_dir is None:
        print(f'Directory ending with {suffix} not found.')
    else:
        return date_dir


def list_dirs(parent_dir_path):
    """
    Returns all child directory names in parent directory as a list.
    Args:
        parent_dir_path (str): path to directory containing subdirectories.
    """
    return [d for d in os.listdir(parent_dir_path) if
            os.path.isdir(pathlib.Path(parent_dir_path, d))]


def read_subdir_paths(parent_dir=''):
    """
    Returns directory paths for all subdirectories in provided directory
    path as strings in a list, and number of subdirectories as an integer.S
    Args:
        parent_dir (str, default=''): directory in which to locate all
            subdirectories, defaults to current working directory if not
            path is provided.
    """
    if parent_dir == '':
        parent_dir = pathlib.Path.cwd()

    # locate subdirectories
    try:
        subdirs = [parent_dir + file for file in list_dirs(parent_dir)]
    except ValueError:
        print('Provided directory path not found.')
    # store directory count as number of recordings
    num_dirs = len(set(subdirs))

    # load each file
    file_paths = []
    for subdir in subdirs:
        file_path, _ = read_file_paths(indir=subdir)
        for file in file_path:
            file_paths.append(file)

    return file_paths, num_dirs


def read_metadata(file_path, num_comment_lines=3):
    """
    Returns metadata stored as comments in the first few lines of a
    BioTracker-generated CSV file as list where each comment is an item
    stored as a string.

    Args:
        file_path (str): path to BioTracker-generated CSV file.
        num_comment_lines (int, default=3): number of lines at the beginning
            of file that contain comments.
    """
    with open(file_path) as file:
        # Read specified number of lines
        metadata = [file.readline() for line in range(num_comment_lines)]

    return metadata


def extract_comments_as_dict(dic):
    """
    Takes list of key-value pair comments and returns dict by 
    splitting on standard python chars # and \n.
    """
    comments = [comment.split('#')[1].strip() for comment in dic]
    keys = [val.split(':')[0] for val in comments]
    keys_comments = dict(zip(keys, comments))
    values = [k.replace(j + ':', '').strip() for j, k in keys_comments.items()]
    dic = dict(zip(keys, values))
    for key in dic.keys():
        dic[key] = dic[key].strip()
        try:
            dic[key] = float(dic[key])
        except ValueError:
            pass

        return dic


def get_chunk_number_from_path(file_path, dir_sep='/', subdir_sep='.',
                               as_str=False):
    """
    Returns last two characters of file path corresponding to chunk number.

    Args:
        file_path (str): path to file.
        dir_sep (str, default='/'): character that separates directories and
            files in file path.
        subdir_sep (str, default='.'): character that further separates
            directories and files in file path.
        as_str (bool, default=True): if as_str=True, chunk number is returned
            as string.
    """
    # split file path up using '/' and '.' separator
    _chunk_num = pathlib.Path(file_path).stem.split(f'{dir_sep}')[-1].split(f'{subdir_sep}')
    try:
        chunk_num = int(_chunk_num[0])
        if as_str is True:
            return str(chunk_num)
        else:
            return chunk_num
    except ValueError:
        print(f'Chunk value {_chunk_num[0]} is not an integer value.')


def save_trajectory_to_csv(df, f_name='', outdir='', metadata='', 
                           extension='.csv', save_msg=True, add_metadata=True, 
                           suffix='_processed'):
    """
    Saves pandas.DataFrame object as a CSV file and prepends any metadata
    provided in a file object.
    Args:
        df (pandas.DataFrame): dataframe to be saved to file.
        metadata (list, str): metdata to be stored to file, can be either
            a string of a list of strings.
        f_name (str): file path.
        outdir: output directory.
        save_msg (bool, default=True): prints message to confirm track
            has been saved if save_msg=True.
        add_metadata (bool, default=True): adds metadata comments to file
            as header if add_metadata=Trueread_file_path.
        suffix (str, default='_processed'): suffix to add to the end of file
            when saving.
    """
    # save DataFrame to CSV
    
    f = pathlib.Path(f_name).stem + suffix + extension
    path = outdir / f
    df.to_csv(path, sep=',', encoding='utf-8',
              index=False)

    # optionally prepend metadata to CSV
    if add_metadata is True:
        prepend_comments_to_csv(path, metadata)

    # optionally confirm saving
    if save_msg:
        if suffix is '_processed':
            print(f'Processed {pathlib.Path(f_name).stem}{extension} and saved file to disk.\n')
        else:
            print(f'Merged {pathlib.Path(f_name).stem}{extension} and saved file to disk.')
    else:
        pass


def prepend_comments_to_csv(file, comments, extension='.csv'):
    """
    Insert a list of strings as a new lines at the beginning of a CSV file.
    """
    # define name of temporary dummy file
    file_path = pathlib.Path(file).parents[0]
    file_name = pathlib.Path(file).stem
    temp_file = file_path / (file_name + '.bak')
    # open given original file in read mode  and dummy file in write mode
    with open(file, 'r') as read_obj, open(temp_file, 'w') as write_obj:
        # iterate over list of comments and write them to dummy file as lines
        for line in comments:
            write_obj.write(line)
        # read lines from original file and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    file.unlink()
    # rename dummy file as the original file
    new_extension = temp_file.with_suffix(extension)
    temp_file.rename(new_extension)

    
def read_yaml_as_dict(path):
    """
    Returns yaml dict values.
    """
    with open(path, 'r') as stream:
        try:
            dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return dic


def dict_to_comments(dic, sep=': '):
    """
    Takes dictionary key, value paris and returns list of comments.
    """
    comments = ['# '+str(k)+sep+str(v)+'\n' for k, v in dic.items()]

    return comments
