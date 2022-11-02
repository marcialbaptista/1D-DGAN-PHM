# -*- coding: utf-8 -*-
"""
main_12_plot_AE_training_logs.py

Plot the accuracies and losses over time of the training process
Logs should be in the folder ./logs

author: Marcia Baptista (git: marcialbaptista)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Program
##################################################

log_file = "./logs/global/AE_logs_history.csv"
df_history = pd.read_csv(log_file)

for i in [1,2,5,6,]:
    column = df_history.columns[i]
    plt.rcParams["text.usetex"] = False
    signal = df_history[column]
    plt.plot(signal, label=column)
    plt.legend()
    plt.tight_layout()
plt.show()

for i in [3, 7]:
    column = df_history.columns[i]
    plt.rcParams["text.usetex"] = False
    signal = df_history[column]
    plt.plot(signal, label=column)
    plt.legend()
    plt.tight_layout()
plt.show()

for i in [4, 8]:
    column = df_history.columns[i]
    plt.rcParams["text.usetex"] = False
    signal = df_history[column]
    plt.plot(signal, label=column)
    plt.legend()
    plt.tight_layout()
plt.show()