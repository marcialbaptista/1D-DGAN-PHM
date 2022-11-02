# -*- coding: utf-8 -*-
"""
main_14_generate_GAN_training_data_per_sensor.py

Generates the training dataset of GAN synthetic signals

author: Marcia Baptista (git: marcialbaptista)
"""

from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from random import random
from numpy.random import normal
import math

##################################################
# Debug parameters (show plots or not)
##################################################

show_synthetic_data = False

##################################################
# Control parameters (feature to synthesize)
##################################################

index_sensor_name = 0

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
# Distribution class
##################################################


class Distribution(object):

    def __init__(self, dist_name, params):
        self.name = dist_name
        self.params = params
        self.is_fitted = True
        self.results = []

    def sample(self, n=1):
        dist_name = self.name
        param = self.params
        # initiate the scipy distribution
        dist = getattr(scipy.stats, dist_name)
        val = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        self.results.append(val)
        return val

    def plot(self, y):
        plt.hist(self.results, alpha=0.5, label='Actual')
        plt.legend(loc='upper right')
        plt.show()


##################################################
# Functions
##################################################


# function to concatenate array of data frames
def read_pandas_array(pd_array):
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)

def func(x, p, b):
    return p * np.power(b*x, pw)


def func_d(x, p, b):
    return -p * np.log(b*x)

##################################################
# Read densities and model parameters
##################################################


# read the data about the density of the inflection points
elbow_point_dists = pd.read_csv('./data_models/elbow_point_density.csv')
dist_name = elbow_point_dists['Distribution'][0]
dist_params = elbow_point_dists['Distribution parameters'][0]
dist_params = eval(dist_params)
ep_distribution = Distribution(dist_name, dist_params)
print('Inflection point distribution:', dist_name, dist_params)

# read the data about the density of the faulty stage duration
faulty_stage_duration_dists = pd.read_csv('./data_models/faulty_stage_duration_density.csv')
dist_name = faulty_stage_duration_dists['Distribution'][0]
dist_params = faulty_stage_duration_dists['Distribution parameters'][0]
dist_params = eval(dist_params)
faulty_stage_duration_distribution = Distribution(dist_name, dist_params)
print('Faulty stage duration distribution:', dist_name, dist_params)

# read the data about the density of the slopes in the nominal stage
print('\nDistributions and parameters for slope (nominal degradation rate):')
slopes_dist_dic = defaultdict(Distribution)
slopes_dists = pd.read_csv('./data_models/slopes_density.csv')
sensor_names = slopes_dists['Sensors']
best_distributions = slopes_dists['Distributions']
params_distributions = slopes_dists['Parameters']
for sensor_name, dist_name, dist_params in zip(sensor_names, best_distributions, params_distributions):
    params = eval(dist_params)
    print("Sensor name:", sensor_name, "DistName:", dist_name, "Params:", params)
    slopes_dist_dic[sensor_name] = Distribution(dist_name, params)

# read the data about the density of the end of life measurements
print('\nDistributions and parameters for end of life (EoL) measurements:')
eol_dist_dic = defaultdict(Distribution)
eol_dists = pd.read_csv('./data_models/EoL_density.csv')
sensor_names = eol_dists['Sensors']
best_distributions = eol_dists['Distributions']
params_distributions = eol_dists['Parameters']
for sensor_name, dist_name, dist_params in zip(sensor_names, best_distributions, params_distributions):
    params = eval(dist_params)
    print("Sensor name:", sensor_name, "DistName:", dist_name, "Params:", params)
    eol_dist_dic[sensor_name] = Distribution(dist_name, params)

# read the data about the density of the end of life measurements
print('\nDistributions and parameters for noise levels:')
noise_dist_dic = defaultdict(Distribution)
noise_dists = pd.read_csv('./data_models/noise_density.csv')
sensor_names = noise_dists['Sensors']
best_distributions = noise_dists['Distributions']
params_distributions = noise_dists['Parameters']
for sensor_name, dist_name, dist_params in zip(sensor_names, best_distributions, params_distributions):
    params = eval(dist_params)
    print("Sensor name:", sensor_name, "DistName:", dist_name, "Params:", params)
    noise_dist_dic[sensor_name] = Distribution(dist_name, params)

##################################################
# Main Program
##################################################

# read the data frame
# all feature names
feature_names = ['unit_number', 'time', 'altitude', 'mach number', 'throttle_resolver_angle',
                 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
# non-flat sensor names
sensor_names = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
data = [ pd.read_csv('./CMAPSS_data/train_FD001.txt', sep='\s+', names=feature_names)]
df = read_pandas_array(data)

# normalize the data
for sensor_name in sensor_names:
    original_signal = np.array(df.loc[:, sensor_name].values)
    df.loc[:, sensor_name] = (original_signal - np.min(original_signal)) / (np.max(original_signal)  - np.min(original_signal))

# figure out if signal has increasing or decreasing trend
increasing_trend_dic = defaultdict(bool)
for sensor_name in sensor_names:
    original_signal = np.array(df.loc[df["unit_number"] == 1, sensor_name].values)
    if original_signal[-1] >= original_signal[0]:
        increasing_trend_dic[sensor_name] = True
    else:
        increasing_trend_dic[sensor_name] = False



for index_sensor_name in range(len(sensor_names)):
    samples = []
    noisy_samples = []
    # number of synthetic units
    nr_samples = 1000

    # Loop for generation of synthetic data
    for sample_index in range(nr_samples):
        print('Generating unit', sample_index)
        inflection_point = max(1, int(ep_distribution.sample()[0]))
        faulty_stage_len = int(faulty_stage_duration_distribution.sample()[0])
        while True:
            sensor_name = sensor_names[index_sensor_name]
            # check if signal is increasing or decreasing
            increasing_trend = increasing_trend_dic[sensor_name]
            # sample the measurement at the end of life
            if increasing_trend:
                final_measurement = eol_dist_dic[sensor_name].sample()[0]
            else:
                final_measurement = math.fabs(eol_dist_dic[sensor_name].sample()[0])
            # sample the noise level
            noise_level = max(0.1, (1/4)*noise_dist_dic[sensor_name].sample()[0])
            signal = []
            # sample and estimate the best slope of the nominal stage
            slope = slopes_dist_dic[sensor_name].sample()[0]
            if increasing_trend:
                slope = min(math.fabs(slope), 0.01)
            else:
                slope = -min(math.fabs(slope), 0.01)
            for point in range(inflection_point):
                signal.append(slope * point)
            # add noise to the synthetic signal (signal)
            noisy_signal = signal + normal(0, noise_level, len(signal))

            if show_synthetic_data:
                plt.plot(signal, color="green", linewidth=3)
                plt.scatter(range(len(noisy_signal)), noisy_signal, color="green", label="Nominal noisy signal")
                plt.axvline(inflection_point, label="Inflection point", linestyle="--", color="black")
                plt.xlabel("Time (cycles)", fontsize=16)

            # power of the exponential function (selected after some experiences)
            pw = random() * 4 + 1.5

            x = [0.0, faulty_stage_len]
            if increasing_trend:
                y = [0, final_measurement - signal[-1]]
            else:
                y = [0, final_measurement - signal[-1]]

            # try several times to find a suitable curve between the measurement at the elbow point
            # and the measurement at the end of life
            run_curve_fit = True
            while run_curve_fit:
                try:
                    pw = random() * 4 + 1.5
                    popt, pcov = curve_fit(func, x, y, maxfev=5000000)
                    run_curve_fit = False
                except Exception as e:
                    print("* Error - curve_fit failed")
                    continue

            xf = np.linspace(0, faulty_stage_len, faulty_stage_len)

            # create the faulty synthetic signals (pure and noisy)
            if increasing_trend:
                signal2 = np.array(func(xf, *popt)) + signal[-1]
                noisy_signal2 = signal2 + normal(0, noise_level, faulty_stage_len)
                if show_synthetic_data:
                    plt.plot(np.array(x) + inflection_point, np.array(y) + signal[-1], 'ko')
                    plt.plot(xf + inflection_point, signal2, color="red", linewidth=3)
                    plt.scatter(np.array(range(len(noisy_signal2))) + inflection_point, noisy_signal2, color="red", label="Faulty noisy signal")
            else:
                signal2 = - np.array(func(xf, *popt)) + signal[-1]
                noisy_signal2 = signal2 + normal(0, noise_level, faulty_stage_len)
                if show_synthetic_data:
                    plt.plot(np.array(x) + inflection_point, -np.array(y) + signal[-1], 'ko')
                    plt.plot(xf + inflection_point, signal2, color="red",
                             linewidth=3)
                    plt.scatter(np.array(range(len(noisy_signal2))) + inflection_point, noisy_signal2, color="red",
                                label="Noisy Signal")
            final_measurement_sgn2 = final_measurement + signal[-1]
            if not increasing_trend:
                final_measurement_sgn2 = -(final_measurement - signal[-1])
            if round(signal2[-1],1) != round(final_measurement_sgn2,1):
                #print("Did not hit last measurement in curve fit", signal2[-1], final_measurement_sgn2)
                continue
            if show_synthetic_data:
                plt.legend()
                plt.show()
            # add some variability to the initial measurement value
            change = 0
            if increasing_trend:
                change = random()
            else:
                change = -random()
            change = 0
            total_signal = np.concatenate((signal, signal2)) + change
            total_noisy_signal = np.concatenate((noisy_signal, noisy_signal2)) + change
            samples.append(total_signal)
            noisy_samples.append(total_noisy_signal)
            if False:
                plt.plot(total_signal)
                plt.scatter(range(len(total_noisy_signal)), total_noisy_signal)
                plt.show()
            break

    ##################################################
    # Writting stage (check folders)
    ##################################################

    # Save data to file for re-use
    np.save('./training_data/' + sensor_name + '/GAN_pure_signals.npy', samples, allow_pickle=True)
    # Save data to file for re-use
    np.save('./training_data/' + sensor_name + '/GAN_noisy_signals.npy', noisy_samples, allow_pickle=True)
