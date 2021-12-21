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
This script was used to check whether or not logarithmizing strength target values improves performance
for the tree-based an ANN-based models.

"""

#%% 0.1 Packages
# general data handling modules
import pandas as pd

# visualization
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# other
import importlib
import os

# inhouse modules
import auxiliary_functions as aux  # repeatedly used functions are outsourced to this module
import data_preprocessing as dp    # data preprocessing module
importlib.reload(dp)  # automatic reload (sometimes, jupyter lab needs a kernel restart instead, if python file changed)
importlib.reload(aux)

#%% 0.2 Variables and constants
# data preparation

# visualization
colors = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", '#AA3377', '#DDAA33', "#BBBBBB", "#CC6677"]
fig_path_name_template = "raw_figures/performance_"
# colormap = ["#0077BB", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]  # https://personal.sron.nl/~pault/
# colormap_desat = [sns.desaturate(c, 0.85) for c in colormap]  # desaturated colormap for plots with wide bars

#%% 1. Import and prepare data
filepath = os.path.dirname(os.path.realpath(__file__))
filename = 'model_parameters_performance.xlsx'

# Import excel sheet
# data = pd.read_excel(os.path.join(filepath, filename),
#                      sheet_name=1,              # indicate sheet index (0-indexing)
#                      header=0,                  # indicate header row (0-indexing)
#                      skiprows=[1],              # skip first non-header row, because it only contains column units (0-indexing)
#                      usecols = 'C:G,I:M,O:AC')  # skip columns considered to be useless in the analysis
data_xgb_cv = pd.read_excel(os.path.join(filepath, filename), sheet_name=4, header=0, skiprows=[0], usecols='A:H')
data_xgb_final = pd.read_excel(os.path.join(filepath, filename), sheet_name=3, header=0, skiprows=[0], usecols='A:H')
#data_ann = pd.read_excel(os.path.join(filepath, filename), sheet_name=5, header=0, skiprows=[0], usecols='G:L')

#%% 2. Process and visualize
s_ = 3  # marker size
capsize_ = .2  # width of error bar
figsize_ = aux.cm2inch(12, 9)  # figure size
dpi_ = 1000  # resolution for pdf export
c1 = 0  # first color
c2 = 9  # second color


# --- (1) XGB freshwater ice plot
fig, ax_ = plt.subplots(figsize=figsize_)
colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE XGB', x='cut-off', hue='log(y)', data=data_xgb_cv.loc[data_xgb_cv.Freshwater == True, :],
              ax=ax_, ci='sd', dodge=0.1, capsize=capsize_, linestyles='-', s=s_)

colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE XGB', x='cut-off', hue='log(y)', data=data_xgb_final.loc[data_xgb_final.Freshwater == True, :],
              ax=ax_, ci=None, linestyles='--', s=s_)
ax_.set_xticklabels(['10', '20', '40', '60', '75', '80', 'none'])

log_y_cv = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) CV')
y_cv = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) CV')
log_y = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) final model', linestyle='--')
y = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) final model', linestyle='--')
plt.legend(handles=[log_y_cv, y_cv, log_y, y], loc='upper left')
ax_.set(ylabel='MAE [MPa]', xlabel='Cut-off [MPa]')
plt.tight_layout(pad=0)  # remove white space around plots

plt.savefig(fig_path_name_template + "XGB_fw.pdf", format='pdf', dpi=dpi_)
plt.show


# --- (2) XGB saltwater ice plot
fig, ax_ = plt.subplots(figsize=figsize_)
colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE XGB', x='cut-off', hue='log(y)', data=data_xgb_cv.loc[data_xgb_cv.Freshwater == False, :],
              ax=ax_, ci='sd', dodge=0.1, capsize=capsize_, linestyles='-', s=s_)

colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE XGB', x='cut-off', hue='log(y)', data=data_xgb_final.loc[data_xgb_final.Freshwater == False, :],
              ax=ax_, ci=None, linestyles='--', s=s_)
ax_.set_xticklabels(['10', '20', '25', 'none'])

log_y_cv = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) CV')
y_cv = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) CV')
log_y = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) final model', linestyle='--')
y = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) final model', linestyle='--')
plt.legend(handles=[log_y_cv, y_cv, log_y, y])
ax_.legend_.remove()
ax_.set(ylabel='MAE [MPa]', xlabel='Cut-off [MPa]')
plt.tight_layout(pad=0)  # remove white space around plots

plt.savefig(fig_path_name_template + "XGB_sw.pdf", format='pdf', dpi=dpi_)
plt.show

# --- (3) ANN freshwater ice plot
fig, ax_ = plt.subplots(figsize=figsize_)
colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE ANN', x='cut-off', hue='log(y)', data=data_xgb_cv.loc[data_xgb_cv.Freshwater == True, :],
              ax=ax_, ci='sd', dodge=0.1, capsize=capsize_, linestyles='-', s=s_)

colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE ANN', x='cut-off', hue='log(y)', data=data_xgb_final.loc[data_xgb_final.Freshwater == True, :],
              ax=ax_, ci=None, linestyles='--', s=s_)
ax_.set_xticklabels(['10', '20', '40', '60', '75', '80', 'none'])
ax_.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0'])
ax_.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

log_y_cv = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) CV')
y_cv = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) CV')
log_y = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) final model', linestyle='--')
y = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) final model', linestyle='--')
plt.legend(handles=[log_y_cv, y_cv, log_y, y])
ax_.legend_.remove()
ax_.set(ylabel='MAE [MPa]', xlabel='Cut-off [MPa]')
plt.tight_layout(pad=0)  # remove white space around plots

plt.savefig(fig_path_name_template + "ANN_fw.pdf", format='pdf', dpi=dpi_)
plt.show

# --- (4) ANN saltwater ice plot
fig, ax_ = plt.subplots(figsize=figsize_)
colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE ANN', x='cut-off', hue='log(y)', data=data_xgb_cv.loc[data_xgb_cv.Freshwater == False, :],
              ax=ax_, ci='sd', dodge=0.1, capsize=capsize_, linestyles='-', s=s_)

colors_ordered = [colors[i] for i in [c1, c2]]
sns.set_palette(sns.color_palette(colors_ordered))
sns.pointplot(y='MAE ANN', x='cut-off', hue='log(y)', data=data_xgb_final.loc[data_xgb_final.Freshwater == False, :],
              ax=ax_, ci=None, linestyles='--', s=s_)
ax_.set_xticklabels(['10', '20', '25', 'none'])
ax_.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
ax_.set_yticklabels(['1.0', '1.5', '2.0', '2.5', '3.0'])

log_y_cv = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) CV')
y_cv = mlines.Line2D([], [], color=colors[c2], marker='o', markersize=5, label='log(y) CV')
log_y = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='non log(y) final model', linestyle='--')
y = mlines.Line2D([], [], color=colors[c1], marker='o', markersize=5, label='log(y) final model', linestyle='--')
plt.legend(handles=[log_y_cv, y_cv, log_y, y])
ax_.legend_.remove()
ax_.set(ylabel='MAE [MPa]', xlabel='Cut-off [MPa]')
plt.tight_layout(pad=0)  # remove white space around plots

plt.savefig(fig_path_name_template + "ANN_sw.pdf", format='pdf', dpi=dpi_)
plt.show

