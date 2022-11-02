# -*- coding: utf-8 -*-
"""
main_02_calc_faulty_stage_duration_density.py

Calculates the density function of the length of the faulty stage (After the elbow point)
 -  Change show_kneedle_plots variable to show the intermediate plots

author: Marcia Baptista (git: marcialbaptista)
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import scipy.stats
from pykalman import KalmanFilter
from kneed import KneeLocator

##################################################
# Debug parameters (show plots or not)
##################################################

# debug parameters
show_kneedle_plots = False

##################################################
# Visualization properties
##################################################

# prepare the matplotlib properties
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['CMU Sans Serif']})
plt.rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
plt.rc('text', usetex=True)

##################################################
# Functions
##################################################


def read_pandas_array(pd_array):
    # function to concatenate array of data frames
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)


def estimate_elbow_kneedle(signal_to_analyse, curve='convex', sensitivity=0.001, direction='increasing'):
    # Algorithm of elbow point detection (KneeLocator) in site: https://github.com/arvkevi/kneed
    total = (signal_to_analyse - np.min(signal_to_analyse)) / (np.max(signal_to_analyse) - np.min(signal_to_analyse))
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=1e-5 * np.eye(2), n_dim_obs=1)
    total = kf.em(total).smooth(total)[0][:, 0]
    kneedle = KneeLocator(range(len(total)), total, S=sensitivity, curve=curve, direction=direction, online=True)
    if kneedle.elbow >= len(total):
        print('Error locating elbow point with Kneedle method')
    print(' * Detected the elbow point at (Kneedle method): ' + str(kneedle.elbow) +
          ' size of ' + str(len(signal_to_analyse)))
    if show_kneedle_plots:
        plt.plot(signal_to_analyse, label='Original signal', color='black')
        plt.legend()
        plt.xlabel('Time (cycles)')
        plt.tight_layout()
        plt.show()
        plt.plot(total, label='Average signal', color='black')
        plt.legend()
        plt.xlabel('Time (cycles)')
        plt.tight_layout()
        plt.show()
        plt.plot(total, label='Average signal', color='black')
        plt.axvline(kneedle.elbow, label='Inflection point', linestyle='--', color='black')
        plt.xlabel('Time (cycles)')
        plt.tight_layout()
        plt.legend()
        plt.show()
    return kneedle.elbow

##################################################
# Program
##################################################


# read the data frame
# all feature names
feature_names = ['unit_number', 'time', 'altitude', 'mach number', 'throttle_resolver_angle',
                 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
# non-flat sensor names
sensor_names = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
data = [pd.read_csv('./CMAPSS_data/train_FD001.txt', sep='\s+', names=feature_names)]
df = read_pandas_array(data)

# standardize the data
for sensor_name in sensor_names:
    original_signal = np.array(df.loc[:, sensor_name].values)
    df.loc[:, sensor_name] = (original_signal - np.mean(original_signal)) / np.std(original_signal)

# Calculate maximum length of a signal
units = np.unique(df['unit_number'])
max_len_signal = 0
for j in range(100):
    unit = j + 1
    signal = np.array(df.loc[df['unit_number'] == unit, 'T24'].values)
    max_len_signal = max(max_len_signal, len(signal))
print('Max length of signal: ', max_len_signal)

# Estimate elbow point locations
elbow_points = []
faulty_stages = []
for unit in units:
    print('Estimating elbow point location of unit', unit)
    orig_signal_all = []
    for sensor_name in sensor_names:
        orig_signal = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
        if show_kneedle_plots:
            plt.plot(orig_signal, label='Original signal', color='black')
            plt.legend()
            plt.xlabel('Time (cycles)')
            plt.tight_layout()
            plt.show()
        if orig_signal[-1] > orig_signal[0]:
            orig_signal = (orig_signal - np.min(orig_signal)) / (np.max(orig_signal) - np.min(orig_signal))
            orig_signal_all.append(orig_signal)

    orig_signal = np.sum(orig_signal_all, axis=0)/len(orig_signal_all)
    if orig_signal[-1] > orig_signal[0]:
        eb_point = estimate_elbow_kneedle(orig_signal, sensitivity=0.01, curve='convex', direction='increasing')
    else:
        eb_point = estimate_elbow_kneedle(orig_signal, sensitivity=0.01, curve='convex', direction='decreasing')
    elbow_points.append(eb_point)
    faulty_stages.append(len(orig_signal) - eb_point)

# Create an index array (x) for data with elbow points
y = np.array(faulty_stages)
x = np.arange(max(y))
size = max(y)
size2 = len(y)

# standardize the elbow point array
sc = StandardScaler()
yy = y.reshape(-1, 1)
sc.fit(yy)
y_std = sc.transform(yy)
y_std = y_std.flatten()

dist_names = ['alpha', 'burr', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'cauchy',
              'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib',
              'f', 'fatiguelife', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic',
              'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic',
              'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm',
              'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone',
              'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell',
              'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw',
              'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular',
              't', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min',
              'weibull_max']

print('Number of tested distributions:', len(dist_names))

# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distributed across all bins
percentile_bins = np.linspace(0, 100, 51)
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions
for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)

    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)

    # Get expected counts in percentile bins
    # This is based on a 'cumulative distribution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                          scale=param[-1])
    expected_frequency = []
    for percentile_bin in range(len(percentile_bins) - 1):
        expected_cdf_area = cdf_fitted[percentile_bin + 1] - cdf_fitted[percentile_bin]
        expected_frequency.append(expected_cdf_area)

    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size2
    cum_expected_frequency = np.cumsum(expected_frequency)
    #print('Expected frequency', cum_expected_frequency)
    ss = np.nansum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    chi_square.append(ss)

# Collate results and sort by goodness of fit (best at top)
results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results  = results.sort_values(['chi_square'])
print(results.head())
dist_names = results['Distribution']
chi_square = results['chi_square']
p_values = results['p_value']

# Report results

print('\nDistributions sorted by goodness of fit:')
print('----------------------------------------')
nr_distributions_sort =10
for name, chi, pvalue in zip(results['Distribution'], results['chi_square'], results['p_value']):
    if pvalue > 0.05:
        mark = " (> 0.05)"
    else:
        mark = " (< 0.05)"
    print(name, "&", round(chi, 2), "&", round(pvalue, 2), mark), "\\\\"
print("\n")

print(results)

# Divide the observed data into 100 bins for plotting (this can be changed)
number_of_bins = 100
bin_cutoffs = np.linspace(np.percentile(y, 0), np.percentile(y, 99), number_of_bins)

# Create the plot
h = plt.hist(y, color='0.75')

# Get the top three distributions from the previous phase
number_distributions_to_plot = 3
dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

# Create an empty list to stroe fitted distribution parameters
parameters = []

# Loop through the distributions ot get line fit and paraemters
line_styles = ['-', '--', ':'	]
select_dist_names = ['Inverse Gamma', 'Inverse Gaussian', 'Log Normal']

# Plot the top 3 best distributions
for proper_dist_name, dist_name, linestyle in zip(select_dist_names, dist_names, line_styles):
    # Set up distribution and store distribution paraemters
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    parameters.append(param)

    # Get line for each distribution (and scale to match observed data)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    scale_pdf = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf_fitted, x)
    pdf_fitted *= scale_pdf

    # Add the line to the plot
    plt.plot(pdf_fitted, label=proper_dist_name, color='black', linestyle=linestyle)

    # Set the plot x axis to contain 99% of the data
    # This can be removed, but sometimes outlier data makes the plot less clear
    plt.gca().set_xlim(0, np.percentile(y, 99))

# Add legend and display plot
plt.grid(linestyle='dotted')
plt.gca().set_xlabel('Faulty stage duration', fontsize=15)
plt.gca().set_ylabel('Frequency', fontsize=15)
plt.tight_layout()
plt.legend()
plt.show()

# Store distribution parameters in a dataframe (this could also be saved)
dist_parameters = pd.DataFrame()
dist_parameters['Distribution'] = (results['Distribution'].iloc[0:number_distributions_to_plot])
dist_parameters['Distribution parameters'] = parameters

# Print parameter results
print('\nDistribution parameters (best in first row):')
print('------------------------')

for index, row in dist_parameters.iterrows():
    print('\nDistribution:', row[0])
    print('Parameters:', row[1])


dist_parameters.to_csv('./data_models/faulty_stage_duration_density.csv')
print('\n* Written ./data_models/faulty_stage_duration_density.csv')
print('* Successful end of script to estimate faulty stage duration distribution')
