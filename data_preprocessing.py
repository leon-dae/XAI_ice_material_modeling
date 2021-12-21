"""
Copyright (c) 2021 -
Leon Kellner, Hamburg University of Technology, Germany
https://www2.tuhh.de/skf/

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
"""

#%%
"""
This file contains all functions necessary for preprocessing of the data.
Different function exist for different downstream analyses.
The data_cleaning() function should precede any other functions.

"""
#%%
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


#%%
def data_cleaning(filepath=os.path.dirname(os.path.realpath(__file__)),
                  filename='data points_v1.16.xlsx',
                  output_filename='data_cleaned.xlsx'):
    """Clean database, agnostic to follow-up use or algorithms.

    Parameters
    ----------
    filepath : str, optional
        Location of database, i.e. excel file `filepath`.
    filename : str, optional
        Name of database file `filename`.
    output_filename : str, optional
        Location of output file after cleaning.


    Returns
    -------
    data : pandas.dataframe

    """
    # Import excel sheet ------------------------------------------------------
    # data = pd.read_excel(os.path.join(filepath, filename),
    #                      sheet_name=1,              # indicate sheet index (0-indexing)
    #                      header=0,                  # indicate header row (0-indexing)
    #                      skiprows=[1],              # skip first non-header row, because it only contains column units (0-indexing)
    #                      usecols = 'C:G,I:M,O:AC')  # skip columns considered to be useless in the analysis
    data = pd.read_excel(os.path.join(filepath, filename), sheet_name=1, header=0, skiprows=[1], usecols='C:G,I:J,O:AC')

    # insert original index as in excel table for later reference. header row + unit row + account for 0-indexing = +3
    data.insert(loc=0, column='original_index', value=data.index + 3)
    data.set_index('original_index', inplace=True)

    # Clean up and filter data set --------------------------------------------
    # --- 1. Formatting
    data.rename(columns=lambda x: x.strip(), inplace=True)            # get rid of trailing white space in column names
    data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)  # get rid of spaces in column names

    # --- 2. rename some column identifiers
    data.rename(columns={'peak_stress': 'sig_p'}, inplace=True)
    data.rename(columns={'eta': 'triaxiality'}, inplace=True)

    # --- 3. data cleaning (outliers, correct data types)
    # - make some columns explicitly categorical, otherwise numerical
    cols_categorical = ['type_test', 'type_behavior', 'type_ice', 'type_water', 'columnar_loading']
    for column in data.columns:
        if column in cols_categorical:
            data[column] = data[column].astype('category')
        else:
            data[column] = pd.to_numeric(data[column], errors='coerce')

    # - clean up behavior types
    # replaces all invalid quantities with NaN categorical, removes all unused categories from the index
    valid_behavior = ['ductile', 'brittle', 'brittle tensile', 'brittle shear', 'c-fault', 'p-fault']
    data['type_behavior'].cat.set_categories(valid_behavior, inplace=True)  # set all categories not in valid_behavior to NaN

    # Test types
    data['type_test'].replace(['extrusion'], ['triaxial compression'], inplace=True)
    data['type_test'] = data['type_test'].astype('category')
    data['type_test'].mask(data.loc[:, 'type_test'].isin(['shear']), inplace=True)

    # brittle types; convert all brittle types to simply brittle
    convert_to_brittle = ['brittle tensile', 'brittle shear', 'c-fault', 'p-fault']
    data['type_behavior'].mask(data.loc[:, 'type_behavior'].isin(convert_to_brittle), 'brittle', inplace=True)

    # - water types; convert all non-saltwater, non-nan values to freshwater
    data['type_water'].mask(~data['type_water'].isin(['s', 'f']) & data['type_water'].notnull(), 'f', inplace=True)

    # - ice types
    data['type_ice'].replace(['c', 'g', 'pc', 'r', 'm', 'uc', 's'], ['columnar', 'granular', 'granular', 'ridge', 'mixed', 'unorient_columnar', 'single_crystal'], inplace=True)
    data['type_ice'] = data['type_ice'].astype('category')  # since replace changes dtype, go back to categorical

    # - clean up numerical types: strain_rate
    data['strain_rate'] = np.log10(data['strain_rate'])     # logarithmize strain_rate with base 10
    data['strain_rate'].mask((data['strain_rate'] < -10) | (data['strain_rate'] > 10), inplace=True)  # set values out of range to nan

    # - triaxiality; replace outliers with nan
    data['triaxiality'].mask((data['triaxiality'] < -20) | (data['triaxiality'] > 1), inplace=True)   # set values out of range to nan

    # - temperature; replace outliers with nan
    data['temperature'].mask((data['temperature'] < -100) | (data['temperature'] > 0), inplace=True)  # set values out of range to nan

    # - geometry; add column with largest dimension
    data['largest_dim'] = data[['width', 'depth', 'length', 'diameter']].max(axis=1)

    # --- 4. save data to excel, if desired
    data.to_excel(output_filename)

    return data


#%%
def data_prep_exploratory(data):
    """Prepare data for exploratory analysis.

    To be run after data_cleaning() function.

    Parameters
    ----------
    data : pandas.dataframe
        Input data.

    Returns
    -------
    df : pandas.dataframe
    """
    # --- 1. specific data cleaning
    # - 1.a. Do not keep unwanted / irrelevant features for exploratory data analysis
    keep_columns = ['sig_1', 'sig_2', 'sig_3', 'type_test', 'sig_p', 'type_behavior', 'strain_rate', 'temperature',
                    'grain_size', 'porosity', 'type_ice', 'type_water',
                    'triaxiality', 'volume', 'largest_dim', 'salinity', 'columnar_loading']
    df = data.loc[:, data.columns.isin(keep_columns)].copy()  # Copy desired data to new dataframe

    df['type_ice'].mask(df['type_ice'].isin(['mixed', 'unorient_columnar', 'single_crystal']), inplace=True)  # Set mixed and unorientated columnar ice to 'nan' since these ice types are not well-defined

    # - 1.b. remove unused categories in all categorical columns
    for column in df.select_dtypes(['category']):
        df[column].cat.remove_unused_categories(inplace=True)  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.remove_unused_categories.html

    return df


#%%
def data_prep_behavior_pred(data, freshwater=True, onehot=False, drop_nan=False, exp_cat=False):
    """Prepare data for behavior prediction (i.e. classification).
    To be run after data_cleaning() function.

    Parameters
    ----------
    X : pandas.dataframe
        Input data.
    freshwater : bool, optional
        Whether or not to use freshwater or saltwater data. True: use freshwater only. False: use salt water only.
    onehot : bool, optional
        Whether or not to one hot encoded categorical data. Alternatively ordinal encoding is used.
    drop_nan : bool, optional
        If True removes all rows that contain any nan.
    exp_cat : bool, optional
        If true takes exp of numerically encoded categorical features.

    Returns
    -------
    X, y, X_display, y_display : pandas.dataframe
    """
    # --- 1. specific data cleaning
    # - 1.a. Do not keep unwanted / irrelevant features for behavior prediction
    keep_columns = ['type_test', 'type_behavior', 'strain_rate', 'temperature',
                    'grain_size', 'porosity', 'type_ice', 'type_water',
                    'triaxiality', 'volume', 'columnar_loading']

    if not freshwater:
        keep_columns.append('salinity')

    X = data.loc[:, data.columns.isin(keep_columns)].copy()   # Copy desired data to new dataframe

    X['type_ice'].mask(X.loc[:, 'type_ice'].isin(['mixed', 'unorient_columnar', 'single_crystal']), inplace=True)  # Remove ice types that are not well-defined or rare
    # data['type_ice'] = data['type_ice'].astype('category')  # since replace changes dtype, go back to categorical

    # - 1.b. remove unused categories in all categorical columns
    for column in X.select_dtypes(['category']):
        X[column].cat.remove_unused_categories(inplace=True)  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.remove_unused_categories.html

    # - 1.c. remove salt or freshwater rows
    X = drop_water_type_rows(X, freshwater=freshwater)

    # - 1.d. drop rows and columns
    X.drop('type_water', axis=1, inplace=True)              # drop whole water column if desired. If fresh or saltwater is removed, only one type remains
    X.dropna(subset=['type_behavior'], inplace=True)        # drop rows with no or nan type_behavior values -> is target
    cols_categorical = list(X.select_dtypes(['category']))  # categorical columns
    X.drop(X[(X.type_test == 'uniaxial tension') | (X.type_test == 'shear')].index, inplace=True)   # drop rows with shear and tensile tests
    print('Uniaxial tension and shear tests removed from data set.')

    # - 1.e. drop all rows that still contain nans, required for some algorithms
    if drop_nan:
        X = drop_nan_rows(X)

    # --- 2. Separate and transform data subsets: predictors and targets, transformed and original data
    # - 2.a. generate target (y) and predictor (X) datasets
    y_display = X['type_behavior'].copy()            # target data for display (without encoding or binarizing)
    y = y_display.copy()                             # target data
    X.drop(['type_behavior'], axis=1, inplace=True)  # drop column target values
    cols_categorical.remove('type_behavior')         # pop target column name from categories list
    X_display = X.copy()                             # predictor data for display (without encoding or binarizing)

    # - 2.b. transform categorical to numerical features
    X = cat_to_num_features(X, drop_nan=drop_nan, onehot=onehot, exp_cat=exp_cat)

    # Binarize labels with specific ordering (brittle = 0, ductile = 1)
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#
    y = pd.Series(preprocessing.label_binarize(y, classes=['brittle', 'ductile'])[:, 0])

    # Alternatively, use labelbinarizer, which can be applied in both directions
    # lb = preprocessing.LabelBinarizer()  # binarize ductile/brittle to 1/0
    # lb.fit(y)  # preserve label binarizer for possible later use, e.g. back transform
    # y = pd.Series(lb.transform(y)[:, 0])  # binarized target data

    print('Binarized behavior: ', y_display.iloc[10], ' = ', y.iloc[10])  # print the numerical equivalence of output categories
    print('Encoded columnar loading: ', X_display.iloc[10, X_display.columns.get_loc('columnar_loading')], ' = ', X.iloc[10, X.columns.get_loc('columnar_loading')])
    print('Encoded type of ice: ', X_display.iloc[10, X_display.columns.get_loc('type_ice')], ' = ', X.iloc[10, X.columns.get_loc('type_ice')])

    return X, y, X_display, y_display


#%%
def data_prep_behavior_analytical(data, freshwater=True, onehot=False, drop_nan=False, exp_cat=False):
    """Prepare data for behavior prediction with Renshaw/Schulson/Golding model
    To be run after data_cleaning() function.

    Parameters
    ----------
    X : pandas.dataframe
        Input data.
    freshwater : bool, optional
        Whether or not to use freshwater or saltwater data. True: use freshwater only. False: use salt water only.
    onehot : bool, optional
        Whether or not to one hot encode categorical data. Alternatively ordinal encoding is used.
    drop_nan : bool, optional
        If True removes all rows that contain any nan.
    exp_cat : bool, optional
        If true takes exp of numerically encoded categorical features.

    Returns
    -------
    X, y, X_display, y_display : pandas.dataframe
    """
    # --- 1. specific data cleaning
    # - 1.a. Do not keep unwanted / irrelevant features for behavior prediction
    keep_columns = ['type_test', 'type_behavior', 'strain_rate', 'temperature',
                    'grain_size', 'porosity', 'type_ice', 'type_water',
                    'triaxiality', 'volume', 'columnar_loading', 'sig_1', 'sig_2', 'sig_3', 'largest_dim']

    if not freshwater:
        keep_columns.append('salinity')

    X = data.loc[:, data.columns.isin(keep_columns)].copy()   # Copy desired data to new dataframe

    # - 1.b. remove unused categories in all categorical columns
    for column in X.select_dtypes(['category']):
        X[column].cat.remove_unused_categories(inplace=True)  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.remove_unused_categories.html

    # - 1.c. remove salt or freshwater rows
    X = drop_water_type_rows(X, freshwater=freshwater)

    # - 1.d. drop rows and columns
    X.drop('type_water', axis=1, inplace=True)              # drop whole water column if desired. If fresh or saltwater is removed, only one type remains
    X.dropna(subset=['type_behavior'], inplace=True)        # drop rows with no or nan type_behavior values -> is target
    cols_categorical = list(X.select_dtypes(['category']))  # categorical columns
    X.drop(X[(X.type_test == 'uniaxial tension') | (X.type_test == 'shear')].index, inplace=True)   # drop rows with shear and tensile tests
    print('Uniaxial tension and shear tests removed from data set.')

    # - 1.e. drop all rows that still contain nans, required for some algorithms
    if drop_nan:
        X = drop_nan_rows(X)

    # --- 2. Separate and transform data subsets: predictors and targets, transformed and original data
    # - 2.a. generate target (y) and predictor (X) datasets
    y_display = X['type_behavior'].copy()            # target data for display (without encoding or binarizing)
    y = y_display.copy()                             # target data
    X.drop(['type_behavior'], axis=1, inplace=True)  # drop column target values
    cols_categorical.remove('type_behavior')         # pop target column name from categories list
    X_display = X.copy()                             # predictor data for display (without encoding or binarizing)

    # - 2.b. transform categorical to numerical features
    X = cat_to_num_features(X, drop_nan=drop_nan, onehot=onehot, exp_cat=exp_cat)

    # Binarize labels with specific ordering (brittle = 0, ductile = 1)
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#
    y = pd.Series(preprocessing.label_binarize(y, classes=['brittle', 'ductile'])[:, 0])

    print('Binarized behavior: ', y_display.iloc[10], ' = ', y.iloc[10])  # print the numerical equivalence of output categories
    print('Encoded columnar loading: ', X_display.iloc[10, X_display.columns.get_loc('columnar_loading')], ' = ', X.iloc[10, X.columns.get_loc('columnar_loading')])
    print('Encoded type of ice: ', X_display.iloc[10, X_display.columns.get_loc('type_ice')], ' = ', X.iloc[10, X.columns.get_loc('type_ice')])

    return X, y, X_display, y_display

#%%
def data_prep_strength_pred(data, freshwater=True, onehot=False, drop_nan=True, drop_outlier=False, exp_cat=False):
    """Prepare data for compressive strength prediction.
    To be run after data_cleaning() function.

    Parameters
    ----------
    X : pandas.dataframe
        Input data.
    freshwater : bool, optional
        Whether or not to use freshwater or saltwater data.
    onehot : bool, optional
        Whether or not to one hot encode categorical data. Alternatively ordinal encoding is used.
    drop_nan : bool, optional
        If True removes all rows that contain any nan.
    drop_outlier : double, optional
        If a number is given, all strength values above this number are removed.
    exp_cat : bool, optional
        If true takes exp of numerically encoded categorical features.

    Returns
    -------
    X, y, X_display, y_display : pandas.dataframe
    """
    # --- specific data cleaning
    # - Do not delete unwanted / irrelevant features for strength prediction
    keep_columns = ['type_test', 'sig_p', 'strain_rate', 'temperature',
                    'grain_size', 'porosity', 'type_ice', 'type_water',
                    'triaxiality', 'volume', 'columnar_loading']

    if not freshwater: keep_columns.append('salinity')

    X = data.loc[:, data.columns.isin(keep_columns)].copy()   # Copy desired data to new dataframe

    X['type_ice'].mask(X.loc[:, 'type_ice'].isin(['mixed', 'unorient_columnar', 'single_crystal']), inplace=True)  # Remove ice types that are not well-defined or rare

    # - remove unused categories in all categorical columns
    for column in X.select_dtypes(['category']):
        X[column].cat.remove_unused_categories(inplace=True)  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.remove_unused_categories.html

    # - remove salt or freshwater rows
    X = drop_water_type_rows(X, freshwater=freshwater)

    # - drop rows and columns
    # - Only compression tests
    X['sig_p'].mask((data['sig_p'] > 0), inplace=True)  # Set values of tension tests to nan (only compression tests desired)
    X['sig_p'] = X['sig_p']*-1          # convert to positive strength values
    if (bool(drop_outlier) is True):    # drop outlier strength values if desired
        X['sig_p'].mask((X['sig_p'] > drop_outlier), inplace=True)
    X.drop('type_water', axis=1, inplace=True)      # drop whole water column; if fresh or saltwater is removed, only one type remains
    X.dropna(subset=['sig_p'], inplace=True)        # drop rows with no or nan sig_p values -> sig_p is target
    X.drop(X[(X.type_test == 'uniaxial tension') | (X.type_test == 'shear')].index, inplace=True)   # drop rows with shear and tensile tests

    # - drop all rows that still contain nans, required for some algorithms
    if drop_nan: X = drop_nan_rows(X)

    # --- Separate and transform data subsets: predictors and targets, transformed and original data
    # - generate target (y) and predictor (X) datasets
    y_display = X['sig_p'].copy()  # create target data vector and reverse sign (it doesn't matter whether target values are positive or negative)
    y = y_display.copy()           # data for display (without encoding or binarizing)
    X.drop('sig_p', axis=1, inplace=True)  # drop target data
    X_display = X.copy()            # data for display (without encoding or binarizing)

    # - transform categorical to numerical features
    X = cat_to_num_features(X, drop_nan=drop_nan, onehot=onehot, exp_cat=exp_cat)

    # hard code idx which has both an ice type and columnar loading for both saltwater and freshwater ice
    # This has only informational purpose and can be omitted
    idx = 46
    if freshwater:
        idx = 2309
    print('Encoded columnar loading: ', X_display.loc[idx, 'columnar_loading'], ' = ', X.loc[idx, 'columnar_loading'])
    print('Encoded type of ice: ', X_display.loc[idx, 'type_ice'], ' = ', X.loc[idx, 'type_ice'])

    return X, y, X_display, y_display


#%%
def drop_water_type_rows(df, freshwater=True):
    """Drop fresh water rows and type_water column.

    Set either type_water value to nan for either freshwater or saltwater samples.
    Remove all rows where type_water = nan.

    Parameters
    ----------
    df : pandas.dataframe
        Input data.

    Returns
    -------
    df : pandas.dataframe
    """
    if freshwater:  # set type_water to NaN if saltwater
        df['type_water'].cat.set_categories(['f'], inplace=True)
    else:           # set type_water to NaN if freshwater
        df['type_water'].cat.set_categories(['s'], inplace=True)
        # df.drop('grain_size', axis=1, inplace=True)  # if only saltwater remains, remove grain size column (there are no grain size measurements for saltwater)
    df.dropna(subset=['type_water'], inplace=True)   # drop rows if type_water is NaN to remove saltwater/freshwater ice
    return df


#%%
def drop_nan_rows(df):
    """Drop rows with any nan values left and check for nans.

    Drop columns which contain many nan values. Then drop all rows that have any nan values left.
    Lastly make sure no nans are left in the data.

    Parameters
    ----------
    df : pandas.dataframe
        Input data.

    Returns
    -------
    df : pandas.dataframe
    """
    if 'grain_size' in df.columns:
        df.drop(['grain_size'], axis=1, inplace=True)  # drop these columns before dropping rows because they contain many nans
    df.drop(['porosity'], axis=1, inplace=True)        # -
    df.dropna(inplace=True)                            # drop other rows if they still contain nans
    assert not df.isnull().values.any()                # make sure no nans are left in the data
    return df


#%%
def scale_num_features(input_df, scaler_df):  # standardize numerical columns
    """Standardize numerical data.

    Remove the mean and scale to unit variance, i.e. z-scoring.
    The score of a sample is computed as z = (x - mean)/standard deviation.

    Alternatively use min max scaling, where z = (x - x_mean)/(x_max-x_mean).

    Parameters
    ----------
    df : pandas.dataframe
        Input data.

    scaler_df : sklearn.preprocessing.MinMaxScaler or
    sklearn.preprocessing.StandardScaler

    Returns
    -------
    df : pandas.dataframe
    """
    df = input_df.copy()
    cols_categorical = list(df.select_dtypes(['category']))    # categorical columns
    df[df.columns.difference(cols_categorical)] = scaler_df.transform(df[df.columns.difference(cols_categorical)])  # https://stackoverflow.com/questions/32032836/select-everything-but-a-list-of-columns-from-pandas-dataframe
    return df


#%%
def fit_scaler_num_features(df, method='minmax'):
    """Create scaler for numerical data.

    Remove the mean and scale to unit variance, i.e. z-scoring.
    The score of a sample is computed as z = (x - mean)/standard deviation.

    Alternatively use min max scaling, where z = (x - x_mean)/(x_max-x_mean).

    Parameters
    ----------
    df : pandas.dataframe
        Input data.

    Returns
    -------
    sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler
    """
    cols_categorical = list(df.select_dtypes(['category']))    # categorical columns
    if method == 'minmax':
        return preprocessing.MinMaxScaler().fit(df[df.columns.difference(cols_categorical)])
    elif method == 'standard':
        return preprocessing.StandardScaler().fit(df[df.columns.difference(cols_categorical)])
    else:
        print('Error! Scaler not implemented. Using minmax scaler instead.')
        return preprocessing.MinMaxScaler().fit(df[df.columns.difference(cols_categorical)])


#%%
def cat_to_num_features(df, drop_nan=False, onehot=False, exp_cat=False):
    """Convert categorical to numerical features via one-hot or ordinal encoding.

    Check if there are no nans in the data or if drop_nan=True (if drop_nan=True any nans
    should have been cleaned before). In the no-nan case, perform ordinal or one-hot encoding
    on all categorical columns.
    In the nan case perform ordinal or one-hot encoding column-wise, ignoring and thereby
    preserving nans.

    Parameters
    ----------
    df : pandas.dataframe
        Input data.
    onehot : bool, optional
        Whether or not to one hot encode categorical data. Alternatively ordinal encoding is used.
    drop_nan : bool, optional
        If True removes all rows that contain any nan. If False preserves nans.
    exp_cat : bool, optional
        If true takes exp of numerically encoded categorical features.

    Returns
    -------
    df : pandas.dataframe
    """
    cols_categorical = list(df.select_dtypes(['category']))    # categorical columns
    if onehot:  # one hot encoding
        if exp_cat: print('L\Exponential (np.exp(data)) one hot encoding not implemented!\n')
        if drop_nan or not df.isnull().values.any():           # no nans allowed or no nans in data
            df_ohe = pd.get_dummies(df[cols_categorical])
            df.drop(cols_categorical, axis=1, inplace=True)    # drop categorical type columns
            df = df.join(df_ohe)    # join one hot encoded data to original data
        else:     # if nans are ok -> one-hot encode but leave nans as such
            for column in cols_categorical:
                col_ohe = pd.get_dummies(df[column], dummy_na=True)         # get dummies for one column, last columne is 0/1 for no nan/nan
                nan_in_row = (col_ohe.iloc[:, -1] == 1).values              # boolean vector if there were any nan in the original column
                col_ohe.loc[nan_in_row, col_ohe.columns[:-1]] = np.nan      # if there was a nan in the original column, set complete one-hot row to nan
                col_ohe.drop(col_ohe.columns[-1], axis=1, inplace=True)     # drop last column with nan/non-nan information
                df = df.join(col_ohe)                                       # join with original data
            df.drop(cols_categorical, axis=1, inplace=True)  # drop categorical type columns in original data
    else:  # ordinal encoding
        if drop_nan or not df.isnull().values.any():  # no nans allowed or no nans in data
            oenc = preprocessing.OrdinalEncoder()
            oenc.fit(df[cols_categorical])
            df[cols_categorical] = oenc.transform(df[cols_categorical])
        else:  # if nans are ok -> label encode but leave nans as such
            for column in cols_categorical:     # labelencode columns one by one to preserve nans
                le = preprocessing.LabelEncoder()
                idx = ~df[column].isna()                # not nan index
                temp_enc_column_name = column + '_enc'  # name of temporary column
                le.fit(df.loc[idx, column])             # fit labelencoder to original column
                df.loc[idx, temp_enc_column_name] = le.transform(df.loc[idx, column])   # transform original column and save to new labelencoded column
                df.drop(column, axis=1, inplace=True)                                   # drop original column
                df.rename(columns={temp_enc_column_name: column}, inplace=True)         # rename new column with name of original column
        if exp_cat:  # ordinal encoding with logarithmized types
            print('\nTaking exp(ordinal encoded cat. data).')
            df[cols_categorical] = np.exp(df[cols_categorical])

    return df


#%%
def transform_target(y, back=False):
    """Logarithmize or take exponent of numerical / double target values.

    Parameters
    ----------
    y : input data
    back :  bool, optional.
        If true return logarithmized data. If false return exp(data).

    Returns
    -------
    y : output data
    """
    if back:                # back transform
        return np.exp(y)
    assert not any(y < 0)   # assert that not any value in y is negative (i.e. all values positive)
    return np.log(y)        # return transformed y
