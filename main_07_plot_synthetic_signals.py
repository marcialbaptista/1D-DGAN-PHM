# -*- coding: utf-8 -*-
"""
main_07_plot_synthetic_signals.py

Plot some charts with synthetic signals

author: Marcia Baptista (git: marcialbaptista)
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
data = [ pd.read_csv('./CMAPSS_data/train_FD001.txt', sep='\s+', names=feature_names)]
df = read_pandas_array(data)

# Load data
y_noisy = np.load('./training_data/GAN_noisy_signals.npy', allow_pickle=True)
y_pure = np.load('./training_data/GAN_pure_signals.npy', allow_pickle=True)
flat_pure_signals = [item for sublist in y_pure for item in sublist]
flat_noisy_signals = [item for sublist in y_noisy for item in sublist]

index_trajectory = 0
for synthetic_unit in range(100):
    for sensor_name in sensor_names:
        noisy_signal = (y_noisy[index_trajectory])
        plt.scatter(range(len(noisy_signal)), noisy_signal)
        pure_signal = (y_pure[index_trajectory])
        plt.plot(pure_signal)
        plt.title("Synthetic trajectory of feature " + sensor_name + " of synthetic unit " + str(synthetic_unit+1))
        plt.show()
        orig_signal = df.loc[df['unit_number'] == synthetic_unit + 1, sensor_name].values
        plt.scatter(range(len(orig_signal)), orig_signal, color="black")
        plt.xlabel("Time (cycles)", fontsize=16)
        plt.title("Real trajectory of feature " + sensor_name + " of unit " + str(synthetic_unit + 1))
        plt.show()
        index_trajectory += 1
