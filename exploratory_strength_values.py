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
Exploratory data analysis of strength data, plots histograms and log-normal fits,
runs several statistical tests for log-normality.

"""

#%% 0.1 Packages
# general data handling modules
import numpy as np
import scipy.stats as ss
from statsmodels.stats.diagnostic import lilliefors

# visualization
from matplotlib import pyplot as plt
import seaborn as sns

# figure export
import tikzplotlib

# other
import importlib

# inhouse modules
import auxiliary_functions as aux   # repeatedly used functions are outsourced to this module
import data_preprocessing as dp     # data preprocessing module
importlib.reload(dp)                # automatic reload (sometimes, jupyter lab needs a kernel restart instead, if python file changed)
importlib.reload(aux)

#%% 0.2 Variables and constants
# data preparation
onehot_ = False         # one-hot encode data y/n -> supposedly worse performance with onehot encoding compared to ordinal encoding
standardize_ = False    # standardize input and target t/f
drop_nan_ = False       # drop all rows that contain nan values after data cleaning t/f, not necessary for XGBoosted trees
drop_outlier_ = 0       # drop outliers in strength values

# code execution
goodness_of_fit = True     # check goodness of fit for normal/log-normal dist. with different tests
compute_outlier = True      # compute different measures of outlier detection

# visualization
colormap = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]  # https://personal.sron.nl/~pault/
colormap_desat = [sns.desaturate(c, 0.85) for c in colormap]  # desaturated colormap for plots with wide bars


#%% 1.0 Import and prepare data
file = 'data points_v1.16.xlsx'
data = dp.data_cleaning(filename=file)  # method agnostic initial data cleaning

# method specific data cleaning & split into encoded and non-encoded data
_, _, X_f, y_f = dp.data_prep_strength_pred(data, freshwater=True, onehot=onehot_,
                                            drop_nan=drop_nan_, drop_outlier=drop_outlier_)

_, _, X_s, y_s = dp.data_prep_strength_pred(data, freshwater=False, onehot=onehot_,
                                            drop_nan=drop_nan_, drop_outlier=drop_outlier_)

num_samples_f, num_samples_s = X_f.shape[0], X_s.shape[0]

y_f_log, y_s_log = np.log(y_f), np.log(y_s)

#%% 1.1 General plotting of strength values

alpha_ = 0.8

# histogram of strength values
sns.distplot(y_f, color=colormap[0], norm_hist=False, kde=False, label='freshwater')
sns.distplot(y_s, color=colormap[3], norm_hist=False, kde=False, label='saltwater')
plt.xlabel('Target strength values [MPa]')
plt.ylabel('Frequency [-]')
plt.legend()
plt.show()

# boxplot of strength values to detect outliers
# sns.boxplot(x=y_s_log, color=colormap[3])
# plt.title('Boxplot of log(data) for saltwater')
# plt.show()

# sns.boxplot(x=y_f_log, color=colormap[0])
# plt.title('Boxplot of log(data) for freshwater')
# plt.show()

#%% 1.2 Calculating outliers
if compute_outlier:
    # Adjusted Hubert whiskers
    # see doi: 10.1016/j.csda.2007.11.008.
    whiskers_f, whiskers_s = aux.adjusted_hubert_whiskers(y_f), aux.adjusted_hubert_whiskers(y_s)
    print('\nAdjusted Hubert whiskers for freshwater, min: ', whiskers_f[0], ' max: ', whiskers_f[1])
    print('\nAdjusted Hubert whiskers for saltwater, min: ', whiskers_s[0], ' max: ', whiskers_s[1])

    # Standard deviation of log(y)
    mu_norm_f, std_norm_f = ss.norm.fit(y_f_log)  # fit normal distributions, return paramters mean and std deviation
    mu_norm_s, std_norm_s = ss.norm.fit(y_s_log)
    print('\nStandard deviation freshwater, 1*sig: ', std_norm_f, ' 2*sig: ', 2*std_norm_f)
    # print('\nStandard deviation of log(y) for freshwater, 1*sig: ', np.exp(std_norm_f), ' 2*sig: ', np.exp(2*std_norm_f))
    print('\nStandard deviation of saltwater, 1*sig: ', std_norm_s, ' 2*sig: ', 2*std_norm_s)
    # print('\nStandard deviation of log(y) for saltwater, 1*sig: ', np.exp(std_norm_s), ' 2*sig: ', np.exp(2*std_norm_s))

#%% 1.3 Check target distribution: log-normal and normal
# --- Fit distributions to the data

# - a. log-normal
sigma_lognorm_f, loc_lognorm_f, scale_f = ss.lognorm.fit(y_f, loc=0)  # https://stackoverflow.com/a/36796249
mu_lognorm_f = np.log(scale_f)  # convert to mu
dist_lognorm_f = ss.lognorm(s=sigma_lognorm_f, scale=scale_f)

sigma_lognorm_s, loc_lognorm_s, scale_s = ss.lognorm.fit(y_s, loc=0)  # https://stackoverflow.com/a/36796249
mu_lognorm_s = np.log(scale_s)  # convert to mu
dist_lognorm_s = ss.lognorm(s=sigma_lognorm_s, scale=scale_s)

# - b. Weibull -> log normal gives better results (visually)
# https://stackoverflow.com/a/33079243/
# shape_wb_f, loc_wb_f, scale_wb_f = ss.weibull_min.fit(y_f, floc=0)
# shape_wb_s, loc_wb_s, scale_wb_s = ss.weibull_min.fit(y_s, floc=0)
# dist_wb_f = ss.weibull_min(shape_wb_f, loc_wb_f, scale_wb_f)
# dist_wb_s = ss.weibull_min(shape_wb_s, loc_wb_s, scale_wb_s)

# - freshwater
# for plot style see https://stackoverflow.com/a/54911944
fig, ax_ = plt.subplots()
ax2 = ax_.twinx()
x_f = np.linspace(0, max(y_f), 1000)
sns.distplot(y_f, color=colormap[0], norm_hist=False, kde=False, ax=ax_, label='Histogram of strength values')
sns.lineplot(x_f, dist_lognorm_f.pdf(x_f), ax=ax2, color=colormap[0], label='Log-normal distribution fit')
ax_.set_ylabel('Frequency')
ax2.set_ylabel('Probability')
ax2.set_ylim(ymin=0)
h1, l1 = ax_.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2)
tikzplotlib.save("./tikzplots/strength_dist_fresh.tex")
plt.show()

# - saltwater
fig, ax_ = plt.subplots()
ax2 = ax_.twinx()
x_s = np.linspace(0, max(y_s), 1000)
sns.distplot(y_s, color=colormap[3], norm_hist=False, kde=False, ax=ax_, label='Histogram of strength values')
sns.lineplot(x_s, dist_lognorm_s.pdf(x_s), ax=ax2, color=colormap[3], label='Log-normal distribution fit')
ax_.set_ylabel('Frequency')
ax2.set_ylabel('Probability')
ax2.set_ylim(ymin=0)
h1, l1 = ax_.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2)
tikzplotlib.save("./tikzplots/strength_dist_salt.tex")
plt.show()


# --- plot fit and original distribution
if goodness_of_fit:


    # --- log normal distribution
    # Kolmogorov-Smirnov test
    print('\n### Kolmogorov-Smirnov test for log-normality')
    ks_result_f, ks_result_s = ss.kstest(y_f, dist_lognorm_f.cdf), ss.kstest(y_s, dist_lognorm_s.cdf)  # https://stats.stackexchange.com/a/57900
    print('\nFreshwater: ', ks_result_f, '\nSaltwater: ', ks_result_s)  # if the p value is > 0.05, then your two samples should be identical and balanced

    # --- normal distribution

    # Anderson-Darling test
    # Tests for the (two-parameter) log-normal distribution can be implemented by transforming the data using a logarithm and using the above test for normality.
    # https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
    print('\n### Anderson-Darling test for normality of log(data), freshwater ice.')
    ad_result = ss.anderson(y_f_log, dist='norm')
    print('Statistic: %.3f' % ad_result.statistic)
    for i in range(len(ad_result.critical_values)):
        sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
        if ad_result.statistic < ad_result.critical_values[i]:
            print('%.3f: %.3f, Data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, Data does not look normal (reject H0)' % (sl, cv))

    print('\n### Anderson-Darling test for normality of log(data), saltwater ice.')
    ad_result = ss.anderson(y_s_log, dist='norm')
    print('Statistic: %.3f' % ad_result.statistic)
    for i in range(len(ad_result.critical_values)):
        sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
        if ad_result.statistic < ad_result.critical_values[i]:
            print('%.3f: %.3f, Data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, Data does not look normal (reject H0)' % (sl, cv))

    # Shapiro-Wilks test for normality
    print('\n### Shapiro-Wilks test for normality of log(data), freshwater ice.')
    stat, p = ss.shapiro(y_f_log)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Data looks normal (fail to reject H0)')
    else:
        print('Data does not look normal (reject H0)')

    print('\n### Shapiro-Wilks test for normality of log(data), saltwater ice.')
    stat, p = ss.shapiro(y_s_log)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Data looks normal (fail to reject H0)')
    else:
        print('Data does not look normal (reject H0)')

    # Lilliefors test (Kolmogorov-Smirnov with estimated location and scale)
    # https://stackoverflow.com/a/22135929/
    print('\n### Lilliefors test for normality of log(data), freshwater ice.')
    ksstat_lillie, pvalue_lillie = lilliefors(y_f_log, dist='norm', pvalmethod='table')
    print('KS statistics = %.3f, p-value = %.3f' % (ksstat_lillie, pvalue_lillie))
    # interpret
    alpha = 0.05
    if pvalue_lillie > alpha:
        print('Data looks normal (fail to reject H0)')
    else:
        print('Data does not look normal (reject H0)')

    print('\n### Lilliefors test for normality of log(data), saltwater ice.')
    ksstat_lillie, pvalue_lillie = lilliefors(y_s_log, dist='norm', pvalmethod='table')
    print('KS statistics = %.3f, p-value = %.3f' % (ksstat_lillie, pvalue_lillie))
    # interpret
    alpha = 0.05
    if pvalue_lillie > alpha:
        print('Data looks normal (fail to reject H0)')
    else:
        print('Data does not look normal (reject H0)')

