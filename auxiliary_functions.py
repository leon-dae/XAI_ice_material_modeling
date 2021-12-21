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
This file contains auxiliary functions for the data analysis.

"""

#%%
def is_list(input_, idx=0):
    """Check if input is list and if so return first entry.
    This is sometimes needed for SHAP analyses for classification with > 2 classes.

    Parameters
    ----------
    input_ : list
    idx : integer, optional

    Returns
    -------
    list[idx] : depends on list, should be array
    """
    if isinstance(input_, list):
        return input_[idx]
    return input_

#%%
def binary_classif_metrics_xgb(model, X_test, y_test):
    """Compute metrics for binary classification.

    Parameters
    ----------
    model : xgboost.train
        Trained booster model.
    X_test : pandas.dataframe
        Testing data input.
    y_test : pandas.dataframe
        Testing data target vector.

    Returns
    -------
    [mcc, acc] : list of metrics. MCC is Matthew's correlation coefficient,
    acc is accuracy.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import matthews_corrcoef
    import xgboost as xgb
    y_predict = model.predict(xgb.DMatrix(X_test))  # predict based on test data
    y_predict = (y_predict > 0.5).astype(int)       # convert probabilities to classes
                                                    # Works this way only for binary classification
    mcc = matthews_corrcoef(y_test, y_predict)      # calculate matthews correlation coefficient
    acc = accuracy_score(y_test, y_predict)         # calculate accuracy
    return [mcc, acc]

#%%
def regression_metrics(y_pred, y_test):
    """Compute metrics for regression.

    Parameters
    ----------
    y_pred : pandas.dataframe
        Predicted data target vector.
    y_test : pandas.dataframe
        Testing data target vector.

    Returns
    -------
    metrics : dict of metrics. MSE is mean squared error,
    RMSE is root mean squared error, MAE is mean absolute error,
    RMSLE is root mean squared log error.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    metrics = {'mse': [], 'rmse': [], 'mae': [], 'rmsle': []}
    metrics['mse'] = mean_squared_error(y_test.values, y_pred)     # mean squared error
    metrics['rmse'] = np.sqrt(metrics['mse'])                      # Root mean squared error
    metrics['mae'] = mean_absolute_error(y_test.values, y_pred)    # mean absolute error
    metrics['rmsle'] = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_test.values))))  # root mean square log error. https://stackoverflow.com/a/47623068
    return metrics

#%%
def adjusted_hubert_whiskers(data):
    """Compute adjusted Hubert whiskers.

    Compute adjusted whiskers for boxplots or detection of potential outliers.
    Original source is: M. Hubert and E. Vandervieren, “An adjusted boxplot for
    skewed distributions,” Computational Statistics & Data Analysis, vol. 52,
    no. 12, pp. 5186–5201, 2008, doi: 10.1016/j.csda.2007.11.008.

    Parameters
    ----------
    data: pandas.series

    Returns
    -------
    [lower whisker, upper whisker] : Adjusted boxplot upper/lower whiskers.
    """
    from statsmodels.stats.stattools import medcouple
    import numpy as np
    mc = medcouple(data)                     # medcouple skewness measure
    quartiles = data.quantile([0.25, 0.75])  # quartiles
    IQR = quartiles[0.75] - quartiles[0.25]  # interquartile range
    if mc >= 0:
        return [quartiles[0.25]-1.5*np.exp(-4*mc)*IQR, quartiles[0.75]+1.5*np.exp(3*mc)*IQR]
    return [quartiles[0.25]-1.5*np.exp(-3*mc)*IQR, quartiles[0.75]+1.5*np.exp(4*mc)*IQR]

#%%
# Converge inches to cm for metric plotting
def cm2inch(*tupl):
    """Convert figure size cm to inch values.

    Convert figure size tuple X*Y from centimeter to inch values.

    Parameters
    ----------
    data: tuple

    Returns
    -------
    tuple
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#%%
def colormap():
    """Creates custom colormap.

    Colors are mostly taken from https://personal.sron.nl/~pault/

    Parameters
    ----------


    Returns
    -------
    cmap : list of hexadecimal color codes
    """
    cmap = ['#0077bb',   # blue
            '#33bbee',   # cyan
            '#009988',   # teal
            '#ee7733',   # orange
            '#cc3311',   # red
            '#ee6677',   # red_bright
            '#bb5566',   # red_toned
            '#ee3377',   # magenta
            '#aa3377',   # purple
            '#bbbbbb',   # grey
            '#332288',   # indigo
            '#ddaa33',   # yellow
            '#000000']   # black
    return cmap
"""
equivalent matlab code
colors = struct('blue', [0, 119, 187]./255, ...
                'cyan', [51, 187, 238]./255, ...
                'teal', [0, 153, 136]./255, ...
                'orange', [238, 119, 51]./255, ...
                'red', [204, 51, 17]./255, ...
                'red_bright', [238, 102, 119]./255, ...
                'red_toned', [187, 85, 102]./255, ...
                'magenta', [238, 51, 119]./255, ...
                'purple', [170, 51, 119]./255, ...
                'grey', [187, 187, 187]./255, ...
                'indigo', [51, 34, 136]./255, ...
                'yellow', [221, 170, 51]./255, ...
                'black', [0,0,0]);
"""
