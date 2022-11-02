# -*- coding: utf-8 -*-
"""
main_13_compare_denoising_models.py

Compare the denoising of the C-MAPSS data with all methods

author: Marcia Baptista (git: marcialbaptista)
"""
from __future__ import absolute_import, division, print_function
from keras.models import Sequential
import scipy
from scipy import signal
from config import ConfigCGAN as config
import cgan as model
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
from collections import defaultdict
from keras.models import model_from_json
import math
import pandas as pd
import numpy as np
import pylab as plt
from sklearn.neural_network import MLPRegressor

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

show_denoising_plots = True

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

plt.rc('font',**{'family':'sans-serif','sans-serif':['CMU Sans Serif']})
plt.rc('font',**{'family':'serif','serif':['CMU Serif']})
plt.rc('text', usetex=True)

def create_auto_encoder():
    # load json and create model
    json_file = open('./models_AE/global/model_AE.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./models_AE/global/model_AE.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return loaded_model


def create_sensor_auto_encoder(sensor_name):
    # load json and create model
    json_file = open('./models_AE/' + sensor_name +  '/model_AE.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./models_AE/' + sensor_name + '/model_AE.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return loaded_model


def resample(x, new_size, kind='linear'):
    f = scipy.interpolate.interp1d(np.linspace(0, 1, len(x)), x, kind)
    return f(np.linspace(0, 1, new_size))


# function to calculate trendability
def calculate_trendability(denoising_model = ''):
    global df, sensor_names, sample_units
    #units = np.unique(df['unit_number'])
    trendability_sensors = defaultdict(float)
    for feature_name in sensor_names:
        trendability_feature = math.inf
        for unit1 in sample_units:
            # obtain signal from dataframe
            smooth_signal1 = df.loc[df['unit_number'] == unit1, feature_name + denoising_model].values
            life_unit1 = len(smooth_signal1)
            for unit2 in sample_units:
                # obtain comparison signal from dataframe
                if unit2 == unit1:
                    continue
                smooth_signal2 = df.loc[df['unit_number'] == unit2, feature_name + denoising_model].values
                life_unit2 = len(smooth_signal2)

                if life_unit2 < life_unit1:
                    smooth_signal2_2 = resample(smooth_signal2, life_unit1)
                    smooth_signal1_2 = smooth_signal1
                elif life_unit2 > life_unit1:
                    smooth_signal2_2 = smooth_signal2
                    smooth_signal1_2 = resample(smooth_signal1, life_unit2)

                rho, pval = scipy.stats.pearsonr(smooth_signal1_2, smooth_signal2_2)
                if math.fabs(rho) < trendability_feature:
                    trendability_feature = math.fabs(rho)

        trendability_sensors[feature_name] = round(trendability_feature, 2)
        print("Trendability", feature_name, trendability_sensors[feature_name])
    return trendability_sensors

# function to calculate prognosability
def calculate_prognosability(denoising_model = ''):
    global df, sensor_names, sample_units
    prognosability_sensors = defaultdict(float)
    for feature_name in sensor_names:
        failure_values = []
        bottom_values = []

        for unit in sample_units:
            # obtain signal from dataframe
            smooth_signal = df.loc[df['unit_number'] == unit, feature_name + denoising_model].values

            smooth_signal = smooth_signal * 1000

            failure_values.append(smooth_signal[-1])
            bottom_values.append(math.fabs(smooth_signal[-1] - smooth_signal[0]))

        prognosability_sensors[feature_name] = round(math.exp(-1 * (np.nanstd(failure_values) / np.nanmean(bottom_values))), 2)
        print("Prognosability", feature_name, prognosability_sensors[feature_name])
    return prognosability_sensors


# function to calculate monotonicity
def calculate_monotonicity(nr_points_slope=10, denoising_model = ''):
    global df, sensor_names, sample_units
    monotonicity_sensors = defaultdict(float)
    for feature_name in sensor_names:
        monotonicity_feature = 0
        for unit in sample_units:
            # obtain signal from dataframe
            signal = df.loc[df['unit_number'] == unit, feature_name + denoising_model].values

            life_unit = len(signal)
            monotonicity_unit = 0

            for index in range(nr_points_slope, life_unit):
                monotonicity_unit += math.copysign(1, signal[index] - signal[index - nr_points_slope]) \
                                     / (life_unit - nr_points_slope)

            monotonicity_feature += math.fabs(monotonicity_unit)
        monotonicity_feature /= (len(sample_units) + 0.0)
        monotonicity_sensors[feature_name] = round(monotonicity_feature,2)
        print("Monotonicity", feature_name, denoising_model, ':', monotonicity_sensors[feature_name])
    return monotonicity_sensors

## function to extend or reduce array with signal
def spline_signal(signal_1D, new_length):
    old_indices = np.arange(0, len(signal_1D))
    new_indices = np.linspace(0, len(signal_1D) - 1, new_length)
    spl = UnivariateSpline(old_indices, signal_1D, k=3, s=0)
    new_signal_1D = spl(new_indices)
    return new_signal_1D

## function to concatenate array of data frames
def read_pandas_array(pd_array):
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)


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
#synthetic_data_noisy = np.load('./GAN_noisy_signals.npy', allow_pickle=True)
#synthetic_data_pure = np.load('./GAN_pure_signals.npy', allow_pickle=True)
#sensor_names = ['T24']

for unit in units:
    for sensor_name in sensor_names:
        df[sensor_name + '_Ks'] = 0
        df[sensor_name + '_MA'] = 0
        df[sensor_name + '_Med'] = 0
        df[sensor_name + '_SG'] = 0
        df[sensor_name + '_AE'] = 0
        df[sensor_name + '_GAN'] = 0
        df[sensor_name + '_GANGlob'] = 0
        df[sensor_name + '_AEGlob'] = 0

increasing_trend_dic = defaultdict(bool)
for sensor_name in sensor_names:
    original_signal = np.array(df.loc[df["unit_number"] == 1, sensor_name].values)
    if original_signal[-1] >= original_signal[0]:
        increasing_trend_dic[sensor_name] = True
    else:
        increasing_trend_dic[sensor_name] = False

class GANDenoiserSpecialized:

    def __init__(self, sensor_name, index_checkpoint):
        self.generator = model.make_generator_model_small()
        self.discriminator = model.make_discriminator_model()
        self.generator_optimizer = tf.optimizers.Adam(config.learning_rate)
        self.discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        #print("Checkpoint file:", tf.train.latest_checkpoint("./models_GAN/global/model/"))
        #checkpoint.restore(tf.train.latest_checkpoint("./models/" + self.name))
        print('./models_GAN/' + 'global' + '/model/ckpt-' + str(index_checkpoint))
        checkpoint.restore('./models_GAN/' + 'global' + '/model/ckpt-' + str(index_checkpoint))

class GANDenoiser:

    def __init__(self, index_checkpoint):
        self.generator = model.make_generator_model_small()
        self.discriminator = model.make_discriminator_model()
        self.generator_optimizer = tf.optimizers.Adam(config.learning_rate)
        self.discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        #print("Checkpoint file:", tf.train.latest_checkpoint("./models_GAN/global/model/"))
        #checkpoint.restore(tf.train.latest_checkpoint("./models/" + self.name))
        print("./models_GAN/global/model/ckpt-" + str(index_checkpoint))
        checkpoint.restore("./models_GAN/global/model/ckpt-" + str(index_checkpoint))

def add_trendability_to_signal(signal, smoothing= 1e-6):
    signal = np.nan_to_num(signal)
    #plt.plot(signal)
    from pykalman import KalmanFilter
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=smoothing * np.eye(2), n_dim_obs=1)
    signal = kf.em(signal).smooth(signal)[0][:, 0]
    #plt.plot(signal, color="red")
    #plt.show()
    if signal[-1]< signal[10]:
        return -np.nan_to_num(signal)
    return np.nan_to_num(signal)

def spline_signal_back_to_original(signal_to_reconstruct, noisy_orig, increasing_trend):
    splined_reconstruction = np.array(signal_to_reconstruct[0]).reshape((config.width * config.height,))
    reconstruction = scipy.signal.decimate(splined_reconstruction, 2)
    reconstructed_signal = spline_signal(reconstruction, len(noisy_orig))

    return reconstructed_signal

GAN_per_sensors = defaultdict(GANDenoiserSpecialized)
#AE_per_sensors = defaultdict(Sequential)
for sensor_name in sensor_names:
    GAN_per_sensors[sensor_name] = GANDenoiserSpecialized(sensor_name, '4')
    #AE_per_sensors[sensor_name] = create_sensor_auto_encoder(sensor_name)

GAN_global = GANDenoiser('3')
AE_global = create_auto_encoder()

sample_units = units[:]
for unit in sample_units:
    print("Unit", unit)
    for sensor_name in sensor_names:
        noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
        increasing_trend = (noisy_orig[-1] - noisy_orig[0] >= 0)
        all_noisy = np.array(df.loc[:, sensor_name].values)
        noisy = (noisy_orig - np.mean(all_noisy)) / (np.std(all_noisy))
        noisy_df = pd.DataFrame(data = {'noisy_signal': np.array(noisy)})

        ma_signal = noisy_df.rolling(10).mean()
        med_signal = noisy_df.rolling(10, win_type='gaussian').mean(std=df.std().mean())
        sg_signal = []
        gan_signal_global = []
        gan_signal = []
        point_index = 10
        increment = 10
        while True:
            noisy_df = pd.DataFrame(data={'noisy_signal': np.array(noisy[:point_index])})
            window_length = point_index
            if window_length % 2 == 0:
                window_length = point_index - 1
            sg_temp_signal = noisy_df.apply(lambda srs: signal.savgol_filter(srs.values, window_length, 3))
            if point_index == 10:
                for i in range(10):
                    sg_signal.append(0)
            else:
                for i in range(increment):
                    sg_signal.append(sg_temp_signal.values[len(sg_temp_signal) - i - 10][0])

            noisy_signal = noisy[:point_index]
            # enlarge/compress the signal to 400 points
            noisy_splined = spline_signal(noisy_signal, config.height * config.width)
            # prepare the data for the convolutional format
            noisy_samples = []
            noisy_sample = (noisy_splined.reshape(config.width, config.height))
            noisy_samples.append(noisy_sample)
            noisy_samples = np.array(noisy_samples).astype('float32')
            noisy_input = noisy_samples.reshape(
                (noisy_samples.shape[0], noisy_samples.shape[1], noisy_samples.shape[2], 1))



            # denoise the data with the deep learning approaches
            predictions_GAN_global = GAN_global.generator(noisy_input, training=False)
            predictions_GAN = GAN_per_sensors[sensor_name].generator(noisy_input, training=False)

            # Model configuration
            width_AE, height_AE = 20, 20
            input_shape = (width_AE, height_AE, 1)

            y_val_noisy_r = []
            noisy_2 = (noisy_orig - np.min(all_noisy)) / (np.max(all_noisy) - np.min(all_noisy))
            noisy_sample = noisy_2[:point_index]
            #noisy_sample = (noisy_sample - np.min(noisy_sample)) / (np.max(noisy_sample) - np.min(noisy_sample))
            noisy_sample = spline_signal(noisy_sample, width_AE * height_AE)
            noisy_sample = noisy_sample.reshape(width_AE, height_AE)
            y_val_noisy_r.append(noisy_sample)
            y_val_noisy_r = np.array(y_val_noisy_r)
            noisy_input_AE = y_val_noisy_r.reshape(
                (int(y_val_noisy_r.shape[0]), y_val_noisy_r.shape[1], y_val_noisy_r.shape[2], 1))

            #predictions_AE_global = (AE_global.predict(noisy_input_AE) * (np.max(noisy) - np.min(noisy)))
            predictions_AE_global = (AE_global.predict(noisy_input_AE)) * (np.max(noisy_orig) - np.min(noisy_orig))
            predictions_AE_global = (predictions_AE_global - np.mean(predictions_AE_global) )/ np.std(predictions_AE_global)
            #predictions_AE = AE_per_sensors[sensor_name][0].predict(noisy[:point_index])

            if point_index == 10:
                gan_signal_global = list(spline_signal_back_to_original(predictions_GAN_global, noisy_signal, increasing_trend))
                gan_signal = list(spline_signal_back_to_original(predictions_GAN, noisy_signal, increasing_trend))
            else:
                for i in range(increment):
                    gan_signal_global.append(spline_signal_back_to_original(predictions_GAN_global, noisy_signal, increasing_trend)[len(sg_temp_signal) - i - 10])
                    gan_signal.append(spline_signal_back_to_original(predictions_GAN, noisy_signal, increasing_trend)[len(sg_temp_signal) - i - 10])

            if point_index >= len(noisy):
                break
            if increment + point_index >= len(noisy) and point_index <= len(noisy):
                increment = len(noisy) - point_index
                point_index = len(noisy)
            else:
                point_index += increment

            # increasing_trend_AE = increasing_trend
            AE_signal_global = spline_signal_back_to_original(predictions_AE_global, noisy_2, increasing_trend)
            #AE_signal = spline_signal_back_to_original(predictions_AE, noisy, increasing_trend_AE)
        #print('len', len(ma_signal), len(gan_signal))
        ma_signal = add_trendability_to_signal(ma_signal.values)
        med_signal= add_trendability_to_signal(med_signal.values)
        sg_signal = add_trendability_to_signal(sg_signal)
        gan_signal = add_trendability_to_signal(gan_signal)
        gan_signal_global = add_trendability_to_signal(gan_signal_global)
        noisy_ks = add_trendability_to_signal(noisy, smoothing = 1e-6)
        df.loc[df['unit_number'] == unit, sensor_name + '_Ks'] = noisy_ks
        df.loc[df['unit_number'] == unit, sensor_name + '_MA'] = ma_signal
        df.loc[df['unit_number'] == unit, sensor_name + '_Med'] = med_signal
        df.loc[df['unit_number'] == unit, sensor_name + '_SG'] = sg_signal
        df.loc[df['unit_number'] == unit, sensor_name + '_GAN'] = gan_signal_global
        df.loc[df['unit_number'] == unit, sensor_name + '_GANGlob'] = gan_signal_global
        #df.loc[df['unit_number'] == unit, sensor_name + '_AE'] = np.nan_to_num(AE_signal)
        df.loc[df['unit_number'] == unit, sensor_name + '_AEGlob'] = np.nan_to_num(AE_signal_global)

        if False and show_denoising_plots:
            fig = plt.figure(figsize=(12, 8))
            if not increasing_trend:
                sg_signal = -sg_signal
                ma_signal = -ma_signal
                med_signal = -med_signal
                gan_signal = -gan_signal
                AE_signal_global = AE_signal_global
            plt.scatter(range(len(noisy)), noisy, color='grey')
            plt.plot(sg_signal, label="Savgol (SG)", linewidth=6)
            plt.plot(ma_signal, label="Moving average (MA)", linewidth=6, linestyle=':')
            plt.plot(med_signal, label="Median (Med)", linestyle='-.', linewidth=6)
            plt.plot(gan_signal, label="GAN", color="brown", linestyle='-', linewidth=6)
            plt.plot(AE_signal_global, label="Autoencoder (DAE)", color="pink", linestyle='--', linewidth=3)
            plt.title(sensor_name)
            plt.legend()
            plt.show()

df.to_csv('results.csv')
print("Monotonicity raw: ")
monotonicities_sensors = calculate_monotonicity(nr_points_slope=10, denoising_model = '')
print("Monotonicity KS: ")
monotonicities_sensors_Ks = calculate_monotonicity(nr_points_slope=10, denoising_model = '_Ks')
print("Monotononicity Moving Average filter: ")
monotonicities_sensors_MA = calculate_monotonicity(nr_points_slope=10, denoising_model = '_MA')
print("Monotononicity Median filter: ")
monotonicities_sensors_Med = calculate_monotonicity(nr_points_slope=10, denoising_model = '_Med')
print("Monotononicity SG filter: ")
monotonicities_sensors_SG = calculate_monotonicity(nr_points_slope=10, denoising_model = '_SG')
print("Monotononicity GAN filter: ")
monotonicities_sensors_GAN = calculate_monotonicity(nr_points_slope=10, denoising_model = '_GAN')
print("Monotononicity GAN global filter: ")
monotonicities_sensors_GANGlob = calculate_monotonicity(nr_points_slope=10, denoising_model = '_GANGlob')
#print("Monotononicity AE filter: ")
#monotonicities_sensors_AE = calculate_monotonicity(nr_points_slope=10, denoising_model = '_AE')
print("Monotononicity AE global filter: ")
monotonicities_sensors_AEGlob = calculate_monotonicity(nr_points_slope=10, denoising_model = '_AEGlob')

print("Trendability raw: ")
trendabilities_sensors = calculate_trendability(denoising_model = '')
print("Trendability Ks: ")
trendabilities_sensors_Ks = calculate_trendability(denoising_model = '_Ks')
print("Trendability GAN filter: ")
trendabilities_sensors_GAN = calculate_trendability(denoising_model = '_GAN')
print("Trendability GAN global filter: ")
trendabilities_sensors_GANGlob = calculate_trendability(denoising_model = '_GANGlob')
print("Trendability MA filter: ")
trendabilities_sensors_MA = calculate_trendability(denoising_model = '_MA')
print("Trendability Median filter: ")
trendabilities_sensors_Med = calculate_trendability(denoising_model = '_Med')
print("Trendability SG filter: ")
trendabilities_sensors_SG = calculate_trendability(denoising_model = '_SG')
#print("Trendability AE filter: ")
#trendabilities_sensors_AE = calculate_trendability(denoising_model = '_AE')
print("Trendability AE global filter: ")
trendabilities_sensors_AEGlob = calculate_trendability(denoising_model = '_AEGlob')

print("Prognosability raw: ")
prognosabilities_sensors = calculate_prognosability(denoising_model = '')
print("Prognosability Ks: ")
prognosabilities_sensors_Ks = calculate_prognosability(denoising_model = '_Ks')
print("Prognosability GAN filter: ")
prognosabilities_sensors_GAN = calculate_prognosability(denoising_model = '_GAN')
print("Prognosability GAN global filter: ")
prognosabilities_sensors_GANGlob = calculate_prognosability(denoising_model = '_GANGlob')
print("Prognosability MA filter: ")
prognosabilities_sensors_MA = calculate_prognosability(denoising_model = '_MA')
print("Prognosability Median filter: ")
prognosabilities_sensors_Med = calculate_prognosability(denoising_model = '_Med')
print("Prognosability SG filter: ")
prognosabilities_sensors_SG = calculate_prognosability(denoising_model = '_SG')
#print("Prognosability AE filter: ")
#prognosabilities_sensors_AE = calculate_prognosability(denoising_model = '_AE')
print("Prognosability AE global filter: ")
prognosabilities_sensors_AEGlob = calculate_prognosability(denoising_model = '_AEGlob')

print('Monotonicities')
for sensor_name in sensor_names:
    print(sensor_name, '&', monotonicities_sensors[sensor_name],'&', monotonicities_sensors_Ks[sensor_name],
          '&', monotonicities_sensors_Med[sensor_name],
          '&', monotonicities_sensors_MA[sensor_name],
          '&', monotonicities_sensors_SG[sensor_name],
          #'&', monotonicities_sensors_AE[sensor_name],
          '&', monotonicities_sensors_AEGlob[sensor_name],
          '&', monotonicities_sensors_GAN[sensor_name],
          '&', monotonicities_sensors_GANGlob[sensor_name],'\\')
print('')

print('Trendabilities')
for sensor_name in sensor_names:
    print(sensor_name, '&',
          trendabilities_sensors[sensor_name],
          '&', trendabilities_sensors_Ks[sensor_name],
          '&', trendabilities_sensors_Med[sensor_name],
          '&', trendabilities_sensors_MA[sensor_name],
          '&', trendabilities_sensors_SG[sensor_name],
          #'&', trendabilities_sensors_AE[sensor_name],
          '&', trendabilities_sensors_AEGlob[sensor_name],
          '&', trendabilities_sensors_GAN[sensor_name],
          '&', trendabilities_sensors_GANGlob[sensor_name],'\\'
          )
print('')

print('Prognosabilities')
for sensor_name in sensor_names:
    print(sensor_name, '&', prognosabilities_sensors[sensor_name],
          '&', prognosabilities_sensors_Ks[sensor_name],
          '&', prognosabilities_sensors_Med[sensor_name],
          '&', prognosabilities_sensors_MA[sensor_name],
          '&', prognosabilities_sensors_SG[sensor_name],
          #'&', prognosabilities_sensors_AE[sensor_name],
          '&', prognosabilities_sensors_AEGlob[sensor_name],
          '&', prognosabilities_sensors_GAN[sensor_name],
          '&', prognosabilities_sensors_GANGlob[sensor_name],'\\')
print('')
import random
random.shuffle(sample_units)
train_units = sample_units[:int(0.6 * len(units))]
test_units = sample_units[int(0.4 * len(units)):]
#for model_name in ['', '_MA', '_Med', '_Ks', '_AE', '_AEGlob', '_GAN', '_GANGlob', '_SG']:
for model_name in ['', '_Ks', '_Med', '_MA',  '_SG', '_AEGlob','_GANGlob']:
    feats = []
    for sensor_name in sensor_names:
        feats.append(sensor_name + model_name)
        for unit in units:
            noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
            len_noisy_orig = len(noisy_orig)
            df.loc[df['unit_number'] == unit, 'RUL'] = len_noisy_orig - np.array(range(len_noisy_orig))
    X_train = df.loc[df['unit_number'].isin(train_units), feats]
    y_train = df.loc[df['unit_number'].isin(train_units), 'RUL'].values
    X_test = df.loc[df['unit_number'].isin(test_units), feats]
    y_test = df.loc[df['unit_number'].isin(test_units), 'RUL'].values
    time_test = df.loc[df['unit_number'].isin(test_units), 'time'].values

    from sklearn.ensemble import RandomForestRegressor
    regr = MLPRegressor(random_state=1, max_iter=100).fit(X_train, y_train)
    #regr = (random_state=1, max_iter=100).fit(X_train, y_train)
    regr = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=2)
    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=2)
    regr_rf.fit(X_train, y_train)

    predictions = regr.predict(X_test)
    mae = np.mean(np.absolute(predictions[time_test > 10] - y_test[time_test > 10]))
    errors = predictions[time_test > 10] - y_test[time_test > 10]
    scoring_errors = []
    false_positives, positives, true_positives = 0, 0, 0
    false_negatives, negatives, true_negatives = 0, 0, 0
    for error in errors:
        if error < 0: # early prediction
            scoring_errors.append(math.exp((math.fabs(error))/10) - 1)

        else: # late prediction
            scoring_errors.append(math.exp((math.fabs(error)) / 13) - 1)
        if error > 0:
            if error > 12:
                false_positives += 1
                positives += 1
            if error < 12:
                positives += 1
                true_positives += 1
        if error < 0:
            if -error > 10:
                false_negatives += 1
                negatives += 1
            if -error < 10:
                negatives += 1
                true_negatives += 1
    scoring_error = np.mean(scoring_errors)
    rmse = np.sqrt(np.mean(np.power(predictions[time_test > 10] - y_test[time_test > 10], 2)))
    print('MAE', model_name, mae, "Std", np.std(errors))
    print('Scoring error', model_name, scoring_error)
    print('RMSE error', model_name, rmse)
    tpr = true_positives / positives
    tnr = true_negatives / negatives

    print('Rates', tpr, tnr)
