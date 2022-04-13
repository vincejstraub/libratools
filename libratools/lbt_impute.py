#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The libratools.lbt_impute module includes methods for detecting and imputing
missing values.
"""


from . import lbt_utils    # local imports


__author__ = "Vincent (Vince) J. Straub"
__email__ = "vincejstraub@gmail.com"
__status__ = "Testing"


def fill_missing(df, remove_first_last=True, interpolate=['time'],
                 ffill=['timeString']):
    """
    Returns DataFrame with NaN values at very end and beginning dropped
    and missing values for columns with constant values interpolated
    using forward fill or linear interpolation.

    Args:
        df (pandas DataFrame): DataFrame to update.
        remove_first_last (bool, default=True): if remove_first_last=True,
            values at the very beginning and end of the DataFrame are dropped.
        interpolate (list, default=['time']): columns which to interpolate
            using linear interpolation.
        ffill (list, default=['timeString']): columns which to interpolate
            using forward fill.

    Returns:
        df (pandas Dataframe).
    """
    # replace NaN values in columns with constant values
    df = df.fillna(value=lbt_utils.get_modal_col_vals(df))

    # interpolate NaN values in time column
    df = interpolate_nan_vals(df, cols=interpolate)

    # forward fill NaN values in timeString col
    df = ffill_col_vals(df, cols=ffill)

    # drop missing values right at the beginning and the end
    if remove_first_last is True:
        df = remove_first_last_nans(df)

    return df


def interpolate_nan_vals(df, cols=['time']):
    """
    Fills NaN values in select DataFrame columns via linear interpolation.
    """
    for col in cols:
        df[col] = df[col].interpolate()

    return df


def ffill_col_vals(df, cols=['timeString']):
    """
    Fills NaN values in select DataFrame columns via forward fill.
    """
    for col in cols:
        df[col] = df[col].ffill()

    return df


def remove_first_last_nans(df, cols=['x', 'y']):
    """
    Removes values at the beginning and end of a DataFrame by using
    first and last valid index for select columns as new index.
    """
    first_idx = df[cols].first_valid_index()
    last_idx = df[cols].last_valid_index()
    df = df.loc[first_idx:last_idx]

    return df
