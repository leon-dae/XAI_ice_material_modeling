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
This script was used to check performance of the empirical approaches compared to
the tree-based regression model.

"""


#%% 0.1 Packages
# general data handling modules
import pandas as pd
import numpy as np

# machine learning
import xgboost

# visualization
from matplotlib import pyplot as plt
import seaborn as sns

# figure export
import tikzplotlib

# other
import importlib
import pickle

# inhouse modules
import auxiliary_functions as aux  # repeatedly used functions are outsourced to this module
import data_preprocessing as dp    # data preprocessing module
importlib.reload(dp)  # automatic reload (sometimes needed)
importlib.reload(aux)

#%% 0.2 Variables and constants
# control variables - data cleaning
freshwater_ = False     # only use freshwater data t/f
drop_nan_ = False       # drop all rows that contain nan values after data cleaning t/f, not necessary for XGBoosted trees
drop_outlier_ = 0       # drop outliers in strength values
if freshwater_: drop_outlier_ = 75.0
transform_y = False     # transform target values for better prediction
exp_cat_ = False
onehot_ = False
only_uniaxial = True   # Drop all observations that are not uniaxial compression

# some general variables for postprocessing
dpi_ = 1000
fig_size_ = aux.cm2inch(12, 10)
fig_path_name_template = "raw_figures/empirical_sw_"
excel_path_name_template = 'excel_outputs/empirical_sw_'
if freshwater_:
    fig_path_name_template = "raw_figures/empirical_fw_"
    excel_path_name_template = 'excel_outputs/empirical_fw_'

file = 'data points_v1.16.xlsx'
#%% 1.0 Preprocessing
# method specific data cleaning & split into encoded and non-encoded data
# Display datasets are not encoded or scaled
data = dp.data_cleaning(filename=file)  # method agnostic initial data cleaning
X, y, X_display, y_display = dp.data_prep_strength_pred(data, freshwater=freshwater_, onehot=onehot_,
                                                        drop_nan=drop_nan_, drop_outlier=drop_outlier_, exp_cat=exp_cat_)

X.to_excel(excel_path_name_template + 'X.xlsx', index=True, header=True)
X_display.to_excel(excel_path_name_template + 'X_display.xlsx', index=True, header=True)

if only_uniaxial:       # drop rows that are not uniaxial
    y.drop(X_display[(X_display.type_test != 'uniaxial compression')].index, inplace=True)
    X.drop(X_display[(X_display.type_test != 'uniaxial compression')].index, inplace=True)
    X_display.drop(X_display[(X_display.type_test != 'uniaxial compression')].index, inplace=True)

# X_newidx = X_display.copy().reset_index()  # use dataframe with resetted index if needed


#%% 2.1 Processing - computing empirical function values
def Jones_2007(strain_rate, type_ice='freshwater'):
    """Compute compressive strength with empirical formula.

    Paper from 2007 doi: 10.1016/j.coldregions.2006.10.

    Parameters
    ----------
    strain_rate : logarithmized strain rate values.

    Returns
    -------
    compressive strength : computed compressive strength values [MPa]
    """
    strain_rate = np.power(10, strain_rate)
    if type_ice == 'freshwater':
        if strain_rate <= 2e-3:
            return 37.8*np.power(strain_rate, 0.216)
        return 8.23*np.power(strain_rate, 0.072)
    if type_ice == 'iceberg':
        if strain_rate <= 2e-3:
            return 24.8*np.power(strain_rate, 0.196)
        return 3.40*np.power(strain_rate, -0.02)


def Ince_2016(temperature):
    """Compute compressive strength with empirical formula.

    Paper from Ince et al. 2016 doi: doi: 10.1080/17445302.2016.1190122.

    Parameters
    ----------
    temperature : Temperature in degrees celsius.

    Returns
    -------
    compressive strength : computed compressive strength values [MPa]
    """
    if freshwater_:
        return -0.35*temperature + 1.65
    return -0.4*temperature + 0.9


def Timco_1990(strain_rate, porosity, type_ice, columnar_loading):
    """Compute compressive strength with empirical formula.

    Paper from Timco and Frederking 1990 doi: 10.1016/S0165-232X(05)80003-5.

    Parameters
    ----------
    strain_rate : logarithmized strain rate values.
    porosity : Total porosity in percent.
    type_ice : Type of as string.
    columnar_loading : type of columnar loading as string.

    Returns
    -------
    compressive strength : computed compressive strength values [MPa]
    """
    strain_rate = np.power(10, strain_rate)
    if strain_rate < 1e-7 or strain_rate >= 1e-3 or np.isnan(porosity):  # return nan if strain rate is outside of valid range
        return np.nan
    if type_ice == 'granular':
        return 49*strain_rate**0.22*(1-np.sqrt(porosity/280))
    if type_ice == 'columnar' or type_ice == 'ridge' and columnar_loading in ['along', 'across']:
        if columnar_loading == 'along':
            return 160*np.power(strain_rate, 0.22)*(1-np.sqrt(porosity/200))  # formular for loading along
        return 37*np.power(strain_rate, 0.22)*(1-np.sqrt(porosity/270))  # formula for loading across
    return np.nan  # if none of the above, e.g. columnar_loading = '45'


def Kovacs_1996(strain_rate, temperature, porosity, salinity):
    """Compute compressive strength with empirical formula.

    Paper from Kovacs 1996 "Sea Ice: Part II. Estimating the Full-Scale Tensile,
    Flexural, and Compressive Strength of First-Year Ice”.

    Parameters
    ----------
    strain_rate : logarithmized strain rate values.
    temperature : Temperature in degrees celsius.
    porosity : Total porosity in percent.
    salinity : Salinity in ppt.

    Returns
    -------
    compressive strength : computed compressive strength values [MPa]
    """
    porosity *= 10  # convert percent to ppt
    strain_rate = np.power(10, strain_rate)
    if strain_rate >= 1e-3 or strain_rate < 1e-6:  # outside of this range, approach is not valid
        return np.nan
    if np.isnan(porosity):  # if porosity not known, estimate with formula
        if not np.isnan(salinity) and temperature < 0.:  # in case data with temperature = 0 exists, this cannot be raised to a negative power
            porosity = 19.37 + 36.18*salinity**(0.91)*abs(temperature)**(-0.69)
        else:  # if salinity is not known, estimation is not possible
            return np.nan
    if porosity <= 25 or porosity > 80:  # if porosity is outside this range, approach is not valid
        return np.nan
    return 2.7e3*strain_rate**(1./3.)*(1./porosity)


# compute compressive strength values
df_plot = pd.DataFrame()
df_plot['y_true'] = np.array(y)

if freshwater_:
    df_plot['Jones'] = np.array([Jones_2007(strain_rate) for strain_rate in X_display.strain_rate])
    df_plot['Ince'] = np.array([Ince_2016(T) for T in X_display.temperature])
else:
    df_plot['Ince'] = np.array([Ince_2016(T) for T in X_display.temperature])
    df_plot['Timco'] = np.zeros(len(y))
    df_plot['Kovacs'] = np.zeros(len(y))
    for idx, row in enumerate(X_display.itertuples(), 0):  # iterate over rows, starting with index=0
        df_plot.loc[idx, 'Timco'] = Timco_1990(row.strain_rate, row.porosity, row.type_ice, row.columnar_loading)
        df_plot.loc[idx, 'Kovacs'] = Kovacs_1996(row.strain_rate, row.temperature, row.porosity, row.salinity)

#%% 2.2 Processing with machine learning model
# https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/
if freshwater_:
    model_path_name_template = 'models/rgsr_xgb_fw_pickle.dat'
else:
    model_path_name_template = 'models/rgsr_xgb_sw_pickle.dat'
model = pickle.load(open(model_path_name_template, "rb"))  # load machine learning model
df_plot['Model'] = model.predict(xgboost.DMatrix(X))

#%% 2.3 Compute absolute error for all approaches
df_ae = pd.DataFrame()
df_ae['Ince'] = abs(df_plot['y_true'] - df_plot['Ince'])
df_ae['Model'] = abs(df_plot['y_true'] - df_plot['Model'])
if freshwater_:
    df_ae['Jones'] = abs(df_plot['y_true'] - df_plot['Jones'])
    plot_order = ['Ince', 'Jones', 'Model']  # order of appearance in swarm- and violinplots
else:
    df_ae['Kovacs'] = abs(df_plot['y_true'] - df_plot['Kovacs'])
    df_ae['Timco'] = abs(df_plot['y_true'] - df_plot['Timco'])
    plot_order = ['Ince', 'Kovacs', 'Timco', 'Model']

print(df_ae.mean())  # get mean absolute error

#%% 3. Postprocessing

# colors
# https://personal.sron.nl/~pault/
colors_rgb = np.array([[0, 119, 187], [51, 187, 238], [0, 153, 136], [238, 119, 51], [204, 51, 17],
                       [238, 51, 119], [170, 51, 119], [221, 170, 51], [187, 187, 187]])
colors_rgb = colors_rgb/255
colors_hex = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", '#AA3377', '#DDAA33', "#BBBBBB"]

# other visualization settings
alpha_ = 0.8 if freshwater_ else 0.6  # higher transparency for saltwater ice (more dots in plot)
s_ = 5
kwargs = dict(alpha=alpha_, s=s_)

# --- Scatterplots
fig, ax_ = plt.subplots()
linspace_ = np.linspace(0, 60) if freshwater_ else np.linspace(0, 35)
plt.plot(linspace_, linspace_, color=colors_hex[8], label='x=y')
if freshwater_:
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Ince'], color=colors_hex[1], **kwargs, label='Ince 2016')
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Jones'], color=colors_hex[0], **kwargs, label='Jones 2007')
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Model'], color=colors_hex[3], **kwargs, label='ML model')
    legend_ = plt.legend()
else:
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Ince'], color=colors_hex[1], **kwargs, label='Ince et al. 2016')
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Timco'], color=colors_hex[2], **kwargs, label='Timco et al. 1990')
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Model'], color=colors_hex[3], **kwargs, label='ML model')
    ax_.scatter(x=df_plot.loc[:, 'y_true'], y=df_plot.loc[:, 'Kovacs'], color=colors_hex[5], s=s_, alpha=1, label='Kovacs 1996')
    ax_.set(xlim=(0, 35))
    handles, labels = ax_.get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[2], handles[4], handles[3]]
    labels = [labels[0], labels[1], labels[2], labels[4], labels[3]]
    legend_ = plt.legend(handles, labels, loc='lower right')

for handle in legend_.legendHandles[1:]:  # edit markes in legend, exclude handle for x=y line by starting at idx = 1
    handle.set_sizes([20.0])    # https://stackoverflow.com/a/43578952
    handle.set_alpha(1)         # https://stackoverflow.com/a/42403471

ax_.ticklabel_format(axis='both', style='plain', useOffset=False)  # no decimals in ticks
ax_.set(ylabel='Predicted strength [MPa]', xlabel='True strength [MPa]')
plt.gcf().set_size_inches(aux.cm2inch(12, 8))
plt.tight_layout(pad=0)  # remove white space around plots
plt.savefig(fig_path_name_template + "scatter.pdf", format='pdf', dpi=dpi_)
if freshwater_:
    tikzplotlib.save("./tikzplots/rgsr_xgb_strength_comp_fw.tex")
else:
    freshwater_: tikzplotlib.save("./tikzplots/rgsr_xgb_strength_comp_sw.tex")

plt.show

# --- Violinplots
# establish correct color ordering for fresh- and saltwater plots
if freshwater_:
    b = [1, 0, 3]
    s_ = 2.5
else:
    b = [1, 5, 2, 3]
    s_ = 1

colors_hex_ordered = [colors_hex[i] for i in b]
sns.set_palette(sns.color_palette(colors_hex_ordered))
fig, ax_ = plt.subplots()
sns.violinplot(data=df_ae, cut=0, inner='quartile', color='white', order=plot_order, ax=ax_, linewidth=1.5)
ax_.set(ylabel='Absolute error [MPa]')
for violin in ax_.collections:  # https://stackoverflow.com/a/62598287/
    violin.set_alpha(0.5)
sns.swarmplot(data=df_ae, size=s_, order=plot_order, zorder=-1, ax=ax_, alpha=1.)

if freshwater_:
    ax_.set_xticklabels(['Ince et al. 2016', 'Jones 2007', 'ML model'])
    ax_.set(xlabel=' ', ylim=(0, 16.5))
else:
    ax_.set_xticklabels(['Ince et al. 2016', 'Kovacs 1996', 'Timco et al. 1990', 'ML model'])
    # plt.setp(ax_.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax_.set(ylim=(0, 23))
    for tick in ax_.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)

plt.gcf().set_size_inches(aux.cm2inch(12, 8))
plt.tight_layout(pad=0)
plt.savefig(fig_path_name_template + "violin.pdf", format='pdf', dpi=dpi_)

if freshwater_:
    tikzplotlib.save("./tikzplots/rgsr_xgb_strength_comp_violin_fw.tex")
else:
    freshwater_: tikzplotlib.save("./tikzplots/rgsr_xgb_strength_comp_violin_sw.tex")
plt.show




