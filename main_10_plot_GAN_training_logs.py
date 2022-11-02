# -*- coding: utf-8 -*-
"""
main_10_plot_GAN_training_logs.py

Plot the accuracies and losses over time of the training process
Adjust the name of the log in the bottom
Logs should be in the folder ./logs

author: Marcia Baptista (git: marcialbaptista)
"""
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
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
# Functions
##################################################

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 1,
        'scalars': 100,
        'tensors': 24,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()
    scalar_accumulators = [event_acc.tensors]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if
                           scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(
        all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key
                                 in keys]

    # Get and validate all steps per key
    all_steps_per_key = [
        [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
        for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(
            all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean(
        [tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events],
        axis=0)
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[tf.make_ndarray(scalar_event.tensor_proto) for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    for key in keys[:len(keys) - 2]:
        values = [val for val in all_per_key[key][2][0]]
        plt.plot(range(0, len(values), 1), values, label=key)
    plt.legend()
    plt.show()

    for key in keys[len(keys) - 2:]:
        values = [val for val in all_per_key[key][2][0]]
        print(values)
        plt.plot(range(0, len(values), 1), values, label=key)
    plt.legend()
    plt.show()

##################################################
# Program
##################################################


if __name__ == '__main__':
    log_file = "./logs/global/events.out.tfevents.1602600323.DESKTOP-H7R9NSB.21460.5.v2"
    plot_tensorflow_log(log_file)

# View training logs as follows in the terminal
# tensorboard --logdir=D:\DenoisingGANProject\logs\global
