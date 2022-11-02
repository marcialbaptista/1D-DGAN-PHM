# -*- coding: utf-8 -*-
"""
main_11_train_AE.py

Train the autoencoder (AE) based on files
 - ./training_data/GAN_noisysignals.npy
 - ./training_data/GAN_puresignals.npy

author: Marcia Baptista (git: marcialbaptista)
"""
from __future__ import absolute_import, division, print_function
__author__ = 'marcia.baptista'
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import math
import pandas as pd
import numpy as np
import pylab as plt
from keras.models import Sequential
from keras.layers import Dropout
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
import data_processing
import pickle


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

##################################################
# Debug parameters (show plots or not)
##################################################

show_synthetic_data = False

##################################################
# Visualization properties
##################################################

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

plt.rc('font',**{'family': 'sans-serif', 'sans-serif': ['CMU Sans Serif']})
plt.rc('font',**{'family': 'serif', 'serif': ['CMU Serif']})
plt.rc('text', usetex=True)

##################################################
# Functions
##################################################


# function to extend or reduce array with signal
def spline_signal(signal_1D, new_length):
    old_indices = np.arange(0, len(signal_1D))
    new_indices = np.linspace(0, len(signal_1D) - 1, new_length)
    spl = UnivariateSpline(old_indices, signal_1D, k=3, s=0)
    new_signal_1D = spl(new_indices)
    return new_signal_1D


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
units = np.unique(df['unit_number'])


# Global data
sensor_indexes = []
y_noisy = []
y_pure = []

# Model configuration
width, height = 20, 20
timesteps = width * height
input_shape = (400, 1)
batch_size = 100  # CHANGE 100, 10
no_epochs = 100 # 400
train_test_split = 0.3
validation_split = 0.3
verbosity = 1
max_norm_value = 1.0



import itertools
for sensor_name in sensor_names:
    print(sensor_name)
    # Reshape data
    y_val_noisy_r = []
    y_val_pure_r = []

    y_noisy_sensor = np.load('./training_data/' + sensor_name + '/GAN_noisy_signals.npy', allow_pickle=True)
    y_pure_sensor = np.load('./training_data/' + sensor_name + '/GAN_pure_signals.npy', allow_pickle=True)
    y_noisy_sensor_reshaped = list(itertools.chain.from_iterable(y_noisy_sensor))
    mean_value = np.mean(y_noisy_sensor_reshaped)
    stdev_value = np.std(y_noisy_sensor_reshaped)
    min_value = np.min(y_noisy_sensor_reshaped)
    max_value = np.max(y_noisy_sensor_reshaped)
    print(min_value, max_value)
    for i in range(len(y_noisy_sensor)):
        noisy_sample = y_noisy_sensor[i]
        pure_sample = y_pure_sensor[i]

        pure_sample = (pure_sample - min_value) / (max_value-min_value)
        noisy_sample = (noisy_sample - min_value) / (max_value-min_value)

        noisy_sample = spline_signal(noisy_sample, width * height)
        pure_sample = spline_signal(pure_sample, width * height)
        y_val_noisy_r.append(noisy_sample)
        y_val_pure_r.append(pure_sample)
    y_val_noisy_r = np.array(y_val_noisy_r)
    y_val_pure_r = np.array(y_val_pure_r)
    noisy_input = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], 1))
    pure_input = y_val_pure_r.reshape((y_val_pure_r.shape[0], y_val_pure_r.shape[1], 1))

    # Train/test split
    percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
    noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
    pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]
    import tensorflow.keras
    from tensorflow.keras.models import Sequential, save_model
    from tensorflow.keras.layers import Conv1D, Conv1DTranspose
    from tensorflow.keras.constraints import max_norm
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    # Create the model
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(Conv1DTranspose(32, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                              kernel_initializer='he_uniform'))
    model.add(Conv1DTranspose(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                              kernel_initializer='he_uniform'))
    model.add(
        Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), padding='same'))

    model.summary()

    # Compile and fit data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy', data_processing.rmse, data_processing.psnr])
    history = model.fit(noisy_input, pure_input, epochs=no_epochs, batch_size=batch_size, validation_split=validation_split)
    cnn = model

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # or save to csv:
    hist_csv_file = './logs/' + sensor_name +'/AE_logs_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # serialize model to JSON
    model_json = cnn.to_json()
    with open('./models_AE/' + sensor_name +'/model_AE.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn.save_weights('./models_AE/' + sensor_name +'/model_AE.h5')
    print("Saved model to disk")