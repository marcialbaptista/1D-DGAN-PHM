# -*- coding: utf-8 -*-
'''
main_03_calc_nominal_degradation_rate_density.py

Calculates the density functions of the nominal
degradation rate (slope) of each feature.

author: Marcia Baptista (git: marcialbaptista)
'''

import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter
import scipy.stats as st
import warnings
import numpy as np

##################################################
# Debug parameters (show plots or not)
##################################################

# debug parameters
show_eol_densities = False

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


# Create models from data
def best_fit_distribution(data_to_analyse, bins=200, ax=None):
    # Model data by finding best fit distribution to data
    # Get histogram of original data
    y, x = np.histogram(data_to_analyse, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    all_distributions = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
        st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l, st.levy_stable,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    sses = []

    # Estimate distribution parameters from data
    for distribution in all_distributions:
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
                    # if axis pass in add to plot

                if sse is not None:
                    sses.append((distribution, sse, params))
        except Exception:
            pass

    sses = sorted(sses, key=lambda tup: tup[1])

    linestyles = ['-', '--', ':']
    index = 0
    for distribution, sse, params in sses[:3]:
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

                try:
                    if ax:
                        if show_eol_densities:
                            pd.Series(pdf, x).plot(ax=ax, color='black', linestyle=linestyles[index],
                                                   label=distribution.name)
                        index += 1
                except Exception:
                    pass
        except Exception:
            pass

    return best_distribution.name, best_params


# function to concatenate array of data frames
def read_pandas_array(pd_array):
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)


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

# standardize the CMAPSS data
for sensor_name in sensor_names:
    original_signal = np.array(df.loc[:, sensor_name].values)
    df.loc[:, sensor_name + '_orig'] = original_signal
    df.loc[:, sensor_name] = (original_signal - np.mean(original_signal)) / np.std(original_signal)

# calculate maximum length of signal
units = np.unique(df['unit_number'])
max_len_signal = 0
for unit in units:
    signal = np.array(df.loc[df['unit_number'] == unit, 'T24'].values)
    max_len_signal = max(max_len_signal, len(signal))
print('Max length of signal: ', max_len_signal)

# Estimate final measurement density
results = pd.DataFrame()
results['Sensors'] = sensor_names
best_distributions = []
best_parameters = []
final_measurements = []
for sensor_name_simulated in sensor_names:
    slopes = []
    intercepts = []
    for unit in units:
        print('Unit', unit)
        # Estimate final measurement of the filtered and normalized (0-1) feature
        print('Sensor name', sensor_name_simulated)
        orig_signal = np.array(df.loc[df['unit_number'] == unit, sensor_name_simulated].values)
        orig_signal_all = np.array(df.loc[:, sensor_name_simulated].values)

        orig_signal2 = (orig_signal - np.min(orig_signal_all)) / (np.max(orig_signal_all) - np.min(orig_signal_all))
        orig_signal2 = orig_signal
        kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                          transition_covariance=0.1 * np.eye(2), n_dim_obs=1)
        filtered_signal = kf.em(orig_signal2).smooth(orig_signal2)[0][:, 0]
        final_measurements.append(filtered_signal[-1])

    # Plot for comparison
    data = pd.Series(np.array(final_measurements))
    ax = None
    if show_eol_densities:
        ax = data.plot(kind='hist', bins=30, density=True, alpha=0.25, label='Actual data', color='black')
        # Save plot limits
        dataYLim = ax.get_ylim()
    dist_str, best_fit_dist_params = best_fit_distribution(data, 200, ax)
    best_distributions.append(dist_str)
    best_parameters.append(best_fit_dist_params)
    if show_eol_densities:
        ax.set_ylim(dataYLim)
        ax.set_xlabel(u'Time (cycles)')
        plt.tight_layout()
        ax.set_ylabel('Frequency')
        plt.legend()
        plt.show()

results['Distributions'] = best_distributions
results['Parameters'] = best_parameters

# Report results
print('\nDistributions and parameters for each feature:')
print('----------------------------------------')
print(results)

results.to_csv('./data_models/EoL_density.csv')
print('\n* Written ./data_models/EoL_density.csv')
print('* Successful end of script to estimate EoL (End of Life) distributions for each feature')
