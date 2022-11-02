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
import cgan as model_cgan
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

show_denoising_plots = False

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
    json_file = open('./models_AE/global3/model_AE.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./models_AE/global3/model_AE.h5")
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
    trendabilities = []
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
        trendabilities.append(trendability_sensors[feature_name])
    print("Trendability", denoising_model, ':', np.mean(trendabilities))
    return trendability_sensors, np.mean(trendabilities)

# function to calculate prognosability
def calculate_prognosability(denoising_model = ''):
    global df, sensor_names, sample_units
    prognosability_sensors = defaultdict(float)
    prognosabilities = []
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
        prognosabilities.append(prognosability_sensors[feature_name])
        print("Prognosability", feature_name, prognosability_sensors[feature_name])
    print("Prognosability", denoising_model, ':', np.mean(prognosabilities))
    return prognosability_sensors, np.mean(prognosabilities)

def calculate_monotonicity_unit(signal):
    life_unit = len(signal)
    monotonicity_unit = 0
    nr_points_slope = 10

    for index in range(nr_points_slope, life_unit):
        monotonicity_unit += math.copysign(1, signal[index] - signal[index - nr_points_slope]) \
                             / (life_unit - nr_points_slope)

    return monotonicity_unit

# function to calculate monotonicity
def calculate_monotonicity(nr_points_slope=10, denoising_model = ''):
    global df, sensor_names, sample_units
    monotonicity_sensors = defaultdict(float)
    monotonicities = []
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
        monotonicities.append(monotonicity_feature)
        print("Monotonicity", feature_name, denoising_model, ':', monotonicity_sensors[feature_name])
    print("Monotonicity", denoising_model, ':', np.mean(monotonicities))
    return monotonicity_sensors, np.mean(monotonicities)

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
        self.generator = model_cgan.make_generator_model_small()
        self.discriminator = model_cgan.make_discriminator_model()
        self.generator_optimizer = tf.optimizers.Adam(config.learning_rate)
        self.discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        ckpt_manager = tf.train.CheckpointManager(self.checkpoint, './models_GAN2/' + sensor_name + '/model', max_to_keep=5)
        self.checkpoint.restore('./models_GAN2/' + sensor_name + '/model/ckpt-' + str(index_checkpoint))
        print("Loaded model from disk" + './models_GAN2/' + sensor_name + '/model/ckpt-' + str(index_checkpoint))


class GANDenoiser:

    def __init__(self, index_checkpoint):
        self.generator = model_cgan.make_generator_model_small()
        self.discriminator = model_cgan.make_discriminator_model()
        self.generator_optimizer = tf.optimizers.Adam(config.learning_rate)
        self.discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        #print("Checkpoint file:", tf.train.latest_checkpoint("./models_GAN/global/model/"))
        #checkpoint.restore(tf.train.latest_checkpoint("./models/" + self.name))
        print("./models_GAN2/global/model/ckpt-" + str(index_checkpoint))
        #checkpoint.restore("./models_GAN/global3/model/ckpt-" + str(index_checkpoint))
        checkpoint.restore("./models_GAN2/global/model/ckpt-" + str(index_checkpoint))

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
    splined_reconstruction = np.array(signal_to_reconstruct).reshape((config.width * config.height,))
    reconstruction = scipy.signal.decimate(splined_reconstruction, 2)
    reconstructed_signal = spline_signal(reconstruction, len(noisy_orig))

    if False and increasing_trend:
        plt.scatter(range(len(reconstructed_signal)), reconstructed_signal)
        plt.scatter(range(len(noisy_orig)), noisy_orig)
        plt.show()

    return reconstructed_signal

GAN_per_sensors = defaultdict(GANDenoiserSpecialized)
AE_per_sensors = defaultdict(Sequential)
sensor_names = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']

indexes_checkpoints = [5, 2, 3, 3, 3, 4, 3, 4, 9, 6, 9, 4, 5, 4]
print(len(indexes_checkpoints))
i = 0
for sensor_name in sensor_names:
    GAN_per_sensors[sensor_name] = GANDenoiserSpecialized(sensor_name, index_checkpoint=indexes_checkpoints[i]) #25
    i+= 1
    AE_per_sensors[sensor_name] = create_sensor_auto_encoder(sensor_name)

GAN_global = GANDenoiser('5')
AE_global = create_auto_encoder()

sensor_names_increasing = []
for sensor_name in sensor_names:
    noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
    increasing_trend = (noisy_orig[-1] > noisy_orig[0])
    sensor_names_increasing.append(sensor_name)



def spline_signal(signal_1D, new_length):
    old_indices = np.arange(0, len(signal_1D))
    new_indices = np.linspace(0, len(signal_1D) - 1, new_length)
    spl = UnivariateSpline(old_indices, signal_1D, k=3, s=0)
    new_signal_1D = spl(new_indices)
    return new_signal_1D

#generate_and_save_images('T24')

plot_images = False
import random
sample_units = units
dict_names = {'_RAW': 'Original', '_Med': 'Med', '_MA': 'MA', '_SG': 'SG', '_AEGlob': 'GDAE', '_AE': 'SDAE', '_GAN': 'SDGAN', '_GANGlob': 'GDGAN'}
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

colors = ['gray', CB91_Blue, CB91_Green, CB91_Purple, CB91_Amber, CB91_Pink, CB91_Blue, CB91_Green]


for unit in sample_units:
    print("Unit", unit)
    continue
    for sensor_name in sensor_names:
        # GAN specialized per sensor feature
        #random.shuffle(sample_units)
        #unit = sample_units[0]
        noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
        increasing_trend = (noisy_orig[-1] > noisy_orig[0])

        ##########################
        # Specific gan
        ##########################
        all_noisy = np.array(df.loc[:, sensor_name].values)

        noisy_minmax = (noisy_orig - np.min(all_noisy)) / (np.max(all_noisy) - np.min(all_noisy))
        if noisy_orig[-1] < noisy_orig[0]:
            noisy_sample = -noisy_orig
            noisy_minmax = (noisy_sample - np.min(-all_noisy)) / (np.max(-all_noisy) - np.min(-all_noisy))

        noisy_minmax_orig = noisy_minmax
        noisy_signal = noisy_minmax


        # enlarge/compress the signal to height x width points
        #noisy_signal = noisy_minmax
        noisy_splined = spline_signal(noisy_signal, config.height * config.width)

        # prepare the data for the convolutional format
        noisy_samples = []
        noisy_sample = (noisy_splined.reshape(config.width, config.height))
        noisy_samples.append(noisy_sample)
        noisy_samples = np.array(noisy_samples).astype('float32')
        noisy_input = noisy_samples.reshape(
            (noisy_samples.shape[0], noisy_samples.shape[1], noisy_samples.shape[2], 1))

        # denoise the data with the generative adversarial network (GAN)
        predictions = GAN_per_sensors[sensor_name].generator(noisy_input, training=False)

        splined_reconstruction = np.array(predictions[0]).reshape((config.width * config.height,))
        reconstruction = scipy.signal.decimate(splined_reconstruction, 2)
        gan_signal = spline_signal(reconstruction, len(noisy_orig))

        noisy_signal = noisy_minmax
        # enlarge/compress the signal to height x width points
        height, width = config.width, config.height
        noisy_splined = spline_signal(noisy_signal, height * width)
        # prepare the data for the convolutional format
        noisy_samples = []
        #noisy_sample = (noisy_splined.reshape(width, height))
        noisy_samples.append(noisy_splined)
        noisy_samples = np.array(noisy_samples).astype('float32')
        noisy_input = noisy_samples.reshape(
            (noisy_samples.shape[0], noisy_samples.shape[1], 1))

        # auto-encoder
        predictions_AE_per_sensor = AE_per_sensors[sensor_name].predict(noisy_input)
        AE_signal_per_sensor = spline_signal_back_to_original(predictions_AE_per_sensor, noisy_minmax, increasing_trend)
        AE_signal_per_sensor = AE_signal_per_sensor #* (np.max(noisy_orig) - np.min(noisy_orig))

        ##########################
        # End of specific gan
        ##########################

        noisy_splined = spline_signal(noisy_minmax_orig, config.height * config.width)
        noisy_test_images = np.array([[noisy_splined]])
        test_inputs = noisy_test_images.reshape(noisy_test_images.shape[0],
                                                config.raw_size,
                                                config.raw_size,
                                                config.channels).astype('float32')
        predictions_AE_global = AE_global.predict(test_inputs)
        GAN_global_predictions = GAN_global.generator(test_inputs, training=False)
        GAN_global_reconstruction = np.array(GAN_global_predictions[0]).reshape((config.height * config.width,))
        # Moving Average (MA) filter
        noisy_df = pd.DataFrame(data={'noisy_signal': np.array(noisy_minmax)})
        ma_signal = noisy_df.rolling(10, min_periods=1).mean()
        ma_signal['noisy_signal'][0:10] = noisy_minmax[0:10]
        # Median (Med) filter
        med_signal = noisy_df.rolling(10,min_periods=1, win_type='gaussian').mean(std=df.std().mean())
        med_signal['noisy_signal'][0:10] = noisy_minmax[0:10]
        # SG filter
        window_length = 11
        sg_signal = noisy_df.apply(lambda srs: signal.savgol_filter(srs.values, window_length, 3))
        for i in range(window_length):
            sg_signal['noisy_signal'][i] = noisy_minmax[i]


        gan_signal_global = list(spline_signal_back_to_original(GAN_global_reconstruction, noisy_minmax, increasing_trend))

        AE_signal_global = spline_signal_back_to_original(predictions_AE_global, noisy_minmax, increasing_trend)

        df.loc[df['unit_number'] == unit, sensor_name + '_RAW'] = noisy_minmax_orig
        df.loc[df['unit_number'] == unit, sensor_name + '_MA'] = ma_signal['noisy_signal'].values
        df.loc[df['unit_number'] == unit, sensor_name + '_Med'] = med_signal['noisy_signal'].values
        df.loc[df['unit_number'] == unit, sensor_name + '_SG'] = sg_signal['noisy_signal'].values
        df.loc[df['unit_number'] == unit, sensor_name + '_GAN'] = gan_signal
        df.loc[df['unit_number'] == unit, sensor_name + '_GANGlob'] = gan_signal_global
        df.loc[df['unit_number'] == unit, sensor_name + '_AEGlob'] = AE_signal_global
        df.loc[df['unit_number'] == unit, sensor_name + '_AE'] = AE_signal_per_sensor


        i = 0
        if plot_images or show_denoising_plots:
            #fig = plt.figure(figsize=(12, 8))
            plt.scatter(range(len(noisy_minmax)), noisy_minmax, color='grey', alpha=0.5)
            for model_name in ['_Med', '_MA',  '_SG']:
                signal_output = df.loc[df['unit_number'] == unit, sensor_name + model_name].values
                mon = math.fabs(calculate_monotonicity_unit(signal_output))
                mon = round(mon,2)
                plt.plot((signal_output), label='Model ' + dict_names[model_name] + ' Mon=' + "%.2f" % mon, lw=2, c=colors[i])
                i += 1
                plt.xlabel('Time (cycles)')
                plt.ylabel(sensor_name)
            plt.title('Unit ' + str(unit))
            plt.legend()
            plt.tight_layout()
            plt.show()


            #fig = plt.figure(figsize=(12, 8))
            #plt.scatter(range(len(noisy_minmax)), noisy_minmax, color='grey', alpha=0.5)

            #for model_name in ['_AE', '_AEGlob', '_GAN', '_GANGlob']:

        if plot_images or show_denoising_plots:
            plt.scatter(range(len(noisy_minmax)), noisy_minmax, color='grey', alpha=0.5)
            for model_name in ['_AE','_AEGlob',  '_GAN', '_GANGlob']:
                signal_output = df.loc[df['unit_number'] == unit, sensor_name + model_name].values
                mon =  math.fabs(calculate_monotonicity_unit(signal_output))
                mon = round(mon, 2)
                plt.plot((signal_output), label='Model ' + dict_names[model_name] + ' Mon=' + "%.2f" % mon, lw=2, c=colors[i])
                plt.xlabel('Time (cycles)')
                plt.ylabel(sensor_name)
                i += 1
            plt.title('Unit ' + str(unit))
            plt.legend()
            plt.tight_layout()
            plt.show()

df = pd.read_csv('results.csv')
x, y, z = [], [], []
if False:
    for unit in sample_units:
        denoised_signal = np.array(df.loc[df['unit_number'] == unit, 'T24_RAW'].values)
        y.append(denoised_signal[-1])
        denoised_signal = np.array(df.loc[df['unit_number'] == unit, 'T24_GAN'].values)
        x.append(denoised_signal[-1])
        denoised_signal = np.array(df.loc[df['unit_number'] == unit, 'T24_MA'].values)
        z.append(denoised_signal[-1])

    print(x)

    plt.hist(x=y, alpha=0.5, bins=10, label='Original', color=CB91_Purple)
    plt.hist(x=x, alpha=0.5, label='SDGAN', color=CB91_Blue)
    plt.hist(x=z, alpha=0.5, label='MA', color=CB91_Green)
    plt.legend(loc='upper right')
    plt.xlabel('EoL Measurement')
    plt.ylabel('Frequency')
    plt.show()


    i = 2
    for model_name in ['_RAW', '_GAN']:
        for sensor_name in ['Ps30']:
            for unit in sample_units:
                denoised_signal = np.array(df.loc[df['unit_number'] == unit, sensor_name+ model_name].values)
                plt.plot(denoised_signal, c=colors[i])
            i += 1
            tren = math.fabs(calculate_trendability(model_name)[0]['Ps30'])
            mon = round(tren, 2)
            title = 'Model ' + dict_names[model_name] + ' Tren=' + "%.2f" % tren
            plt.title(title)
            plt.xlabel('Time (cycles)')
            plt.ylabel('Ps30')
            plt.show()



#df = pd.read_csv('results.csv')
#df.to_csv('results.csv')

if False:
    if True:
        print("Monotonicity raw: ")
        monotonicities_sensors, monotonicities_raw = calculate_monotonicity(nr_points_slope=10, denoising_model = '_RAW')
        print("Monotononicity Moving Average filter: ")
        monotonicities_sensors_MA, monotonicities_MA = calculate_monotonicity(nr_points_slope=10, denoising_model = '_MA')
        print("Monotononicity Median filter: ")
        monotonicities_sensors_Med, monotonicities_Med = calculate_monotonicity(nr_points_slope=10, denoising_model = '_Med')
        print("Monotononicity SG filter: ")
        monotonicities_sensors_SG, monotonicities_SG = calculate_monotonicity(nr_points_slope=10, denoising_model = '_SG')
        print("Monotononicity GAN filter: ")
        monotonicities_sensors_GAN, monotonicities_GAN = calculate_monotonicity(nr_points_slope=10, denoising_model = '_GAN')
        print("Monotononicity GAN global filter: ")
        monotonicities_sensors_GANGlob, monotonicities_GANGlob = calculate_monotonicity(nr_points_slope=10, denoising_model = '_GANGlob')
        print("Monotononicity AE filter: ")
        monotonicities_sensors_AE, monotonicities_AE = calculate_monotonicity(nr_points_slope=10, denoising_model = '_AE')
        print("Monotononicity AE global filter: ")
        monotonicities_sensors_AEGlob, monotonicities_AEGlob = calculate_monotonicity(nr_points_slope=10, denoising_model = '_AEGlob')



    if True:
        print("Trendability raw: ")
        trendabilities_sensors, trendabilities_Raw = calculate_trendability(denoising_model = '_RAW')
        print("Trendability GAN filter: ")
        trendabilities_sensors_GAN, trendabilities_GAN = calculate_trendability(denoising_model = '_GAN')
        print("Trendability GAN global filter: ")
        trendabilities_sensors_GANGlob, trendabilities_GANGlob = calculate_trendability(denoising_model = '_GANGlob')
        print("Trendability MA filter: ")
        trendabilities_sensors_MA, trendabilities_MA = calculate_trendability(denoising_model = '_MA')
        print("Trendability Median filter: ")
        trendabilities_sensors_Med, trendabilities_Med = calculate_trendability(denoising_model = '_Med')
        print("Trendability SG filter: ")
        trendabilities_sensors_SG, trendabilities_SG = calculate_trendability(denoising_model = '_SG')
        print("Trendability AE filter: ")
        trendabilities_sensors_AE, trendabilities_AE = calculate_trendability(denoising_model = '_AE')
        print("Trendability AE global filter: ")
        trendabilities_sensors_AEGlob, trendabilities_AEGlob = calculate_trendability(denoising_model = '_AEGlob')

    if True:
        print("Prognosability raw: ")
        prognosabilities_sensors, prognosabilities_Raw = calculate_prognosability(denoising_model = '_RAW')
        print("Prognosability GAN filter: ")
        prognosabilities_sensors_GAN, prognosabilities_GAN = calculate_prognosability(denoising_model = '_GAN')
        print("Prognosability GAN global filter: ")
        prognosabilities_sensors_GANGlob, prognosabilities_GANGlob = calculate_prognosability(denoising_model = '_GANGlob')
        print("Prognosability MA filter: ")
        prognosabilities_sensors_MA, prognosabilities_MA = calculate_prognosability(denoising_model = '_MA')
        print("Prognosability Median filter: ")
        prognosabilities_sensors_Med, prognosabilities_Med = calculate_prognosability(denoising_model = '_Med')
        print("Prognosability SG filter: ")
        prognosabilities_sensors_SG, prognosabilities_SG = calculate_prognosability(denoising_model = '_SG')
        print("Prognosability AE filter: ")
        prognosabilities_sensors_AE,  prognosabilities_AE = calculate_prognosability(denoising_model = '_AE')
        print("Prognosability AE global filter: ")
        prognosabilities_sensors_AEGlob, prognosabilities_AEGlob = calculate_prognosability(denoising_model = '_AEGlob')

    print('Monotonicities')
    for sensor_name in sensor_names:
        print(sensor_name, '&', "%.2f" %monotonicities_sensors[sensor_name],'&',
              '&', "%.2f" %monotonicities_sensors_Med[sensor_name],
              '&', "%.2f" %monotonicities_sensors_MA[sensor_name],
              '&', "%.2f" %monotonicities_sensors_SG[sensor_name],
              '&', "%.2f" %monotonicities_sensors_AE[sensor_name],
              '&', "%.2f" %monotonicities_sensors_AEGlob[sensor_name],
              '&', "%.2f" %monotonicities_sensors_GAN[sensor_name],
              '&', "%.2f" %monotonicities_sensors_GANGlob[sensor_name],'\\\\')
    if True:
        print('Average:', "%.2f" %round(monotonicities_raw, 2), '&',
                  "%.2f" %round(monotonicities_Med, 2),'&',
                  "%.2f" %round(monotonicities_MA, 2), '&',
                  "%.2f" %round(monotonicities_SG, 2), '&',
                  "%.2f" %round(monotonicities_AE, 2),'&',
                  "%.2f" %round(monotonicities_AEGlob, 2),'&',
                  "%.2f" %round(monotonicities_GAN, 2),'&',
                  "%.2f" %round(monotonicities_GANGlob, 2)
                  )
        print('')

    print('Trendabilities')
    for sensor_name in sensor_names:
        print(sensor_name, '&',
              "%.2f" %trendabilities_sensors[sensor_name],
              '&', "%.2f" %trendabilities_sensors_Med[sensor_name],
              '&', "%.2f" %trendabilities_sensors_MA[sensor_name],
              '&', "%.2f" %trendabilities_sensors_SG[sensor_name],
              '&', "%.2f" %trendabilities_sensors_AE[sensor_name],
              '&', "%.2f" %trendabilities_sensors_AEGlob[sensor_name],
              '&', "%.2f" %trendabilities_sensors_GAN[sensor_name],
              '&', "%.2f" %trendabilities_sensors_GANGlob[sensor_name],'\\\\'
              )
    if True:
        print('Average:', "%.2f" %round(trendabilities_Raw, 2), '&',
              "%.2f" %round(trendabilities_Med, 2),'&',
              "%.2f" %round(trendabilities_MA, 2), '&',
              "%.2f" %round(trendabilities_SG, 2), '&',
              "%.2f" %round(trendabilities_AE, 2),'&',
              "%.2f" %round(trendabilities_AEGlob, 2),'&',
              "%.2f" %round(trendabilities_GAN, 2),'&',
              "%.2f" %round(trendabilities_GANGlob, 2)
              )
        print('')

    print('Prognosabilities')
    for sensor_name in sensor_names:
        print(sensor_name, '&', "%.2f" %prognosabilities_sensors[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_Med[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_MA[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_SG[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_AE[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_AEGlob[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_GAN[sensor_name],
              '&', "%.2f" %prognosabilities_sensors_GANGlob[sensor_name],'\\\\')
    if True:
        print('Average:', "%.2f" % round(prognosabilities_Raw, 2), '&',
              "%.2f" % round(prognosabilities_Med, 2), '&',
              "%.2f" %round(prognosabilities_MA, 2), '&',
              "%.2f" %round(prognosabilities_SG, 2), '&',
              "%.2f" %round(prognosabilities_AE, 2), '&',
              "%.2f" % round(prognosabilities_AEGlob, 2), '&',
              "%.2f" %round(prognosabilities_GAN, 2), '&',
              "%.2f" %round(prognosabilities_GANGlob, 2))
        print('')
    import random

random.seed(8)
random.shuffle(sample_units)

MAEdict = {
    "_RAW": [],
    "_Med": [],
    "_MA": [],
    "_SG": [],
    "_AE": [],
    "_AEGlob": [],
    "_GAN": [],
    "_GANGlob": [],
}
RMSEdict = {
    "_RAW": [],
    "_Med": [],
    "_MA": [],
    "_SG": [],
    "_AE": [],
    "_AEGlob": [],
    "_GAN": [],
    "_GANGlob": [],
}
TPRdict = {
    "_RAW": [],
    "_Med": [],
    "_MA": [],
    "_SG": [],
    "_AE": [],
    "_AEGlob": [],
    "_GAN": [],
    "_GANGlob": [],
}
TNRdict = {
    "_RAW": [],
    "_Med": [],
    "_MA": [],
    "_SG": [],
    "_AE": [],
    "_AEGlob": [],
    "_GAN": [],
    "_GANGlob": [],
}
Score08dict = {
    "_RAW": [],
    "_Med": [],
    "_MA": [],
    "_SG": [],
    "_AE": [],
    "_AEGlob": [],
    "_GAN": [],
    "_GANGlob": [],
}

print('best features')
from sklearn.feature_selection import SelectKBest, chi2
def best_features():
    for sensor_name in sensor_names:
        for unit in units:
            noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
            len_noisy_orig = len(noisy_orig)
            df.loc[df['unit_number'] == unit, 'RUL'] = len_noisy_orig - np.array(range(len_noisy_orig))

    for model_name in ['_RAW', '_Med', '_MA', '_SG', '_AE', '_AEGlob', '_GAN', '_GANGlob']:
        feats = []
        for sensor_name in sensor_names:
            feats.append(sensor_name + model_name)

        X = df[feats]
        y = df['RUL']
        X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
        print(model_name)

        select = SelectKBest(score_func=chi2, k=10)
        z = select.fit_transform(X, y)
        filter = select.get_support()
        features = np.array(feats)

        print("All features:")
        print(features)

        print("Selected best 10:")
        print(features[filter])



for i in range(0,100, 10):
    test_units = sample_units[i:i + 10]
    train_units = mask = sample_units[~np.isin(sample_units, test_units)]
    m = 0
    #for model_name in ['', '_MA', '_Med', '_Ks', '_AE', '_AEGlob', '_GAN', '_GANGlob', '_SG']:
    #print(list(df.columns))
    for model_name in ['_RAW', '_Med', '_MA',  '_SG', '_AE', '_AEGlob','_GAN', '_GANGlob']:
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
        unit_test = df.loc[df['unit_number'].isin(test_units), 'unit_number'].values

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import LinearSVR
        #regr = MLPRegressor(random_state=1, max_iter=100).fit(X_train, y_train)
        #regr = (random_state=1, max_iter=100).fit(X_train, y_train)
        regr = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=2)
        #regr = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=2)
        #regr = SVR(C=1.0, epsilon=0.2)
        regr = LinearSVR()
        regr = MLPRegressor()
        regr.fit(X_train, y_train)
        predictions = regr.predict(X_test)

        rul_unit = y_test[unit_test == test_units[0]]
        predictions_unit = predictions[unit_test == test_units[0]]
        time_unit = time_test[unit_test == test_units[0]]
        acc_unit, acc_unit_10 = 0, 0
        for prediction, rul, time in zip(predictions_unit, rul_unit, time_unit):
            if math.fabs(prediction - rul) <= 0.2 * rul:
                acc_unit += 1
            if math.fabs(prediction - rul) <= 0.2 * rul and time <= 10:
                acc_unit_10 += 1
        acc_unit = np.round(acc_unit / len(predictions_unit)*100,1)
        acc_unit_10 = np.round(acc_unit_10 / len(time_unit[time_unit <= 10])*100,1)

        plt.scatter(time_unit, predictions_unit, label=dict_names[model_name] + ' Acc:' +  '%.1f' % acc_unit + ', ' +  '%.1f' % acc_unit_10, color=colors[m])
        m += 1
        plt.plot(time_unit, rul_unit, color='black')
        plt.plot(time_unit, rul_unit*0.8, color='gray')
        plt.plot(time_unit, rul_unit*1.2, color='gray')
        if model_name == '_SG' or model_name == '_GANGlob':
            plt.xlabel('Time (cycles)')
            plt.ylabel('Predicted RUL (cycles)')
            plt.tight_layout()
            plt.legend()
            plt.savefig('pics/' + model_name + str(test_units[0]) + '_MLP.png')
            plt.close()

        mae = np.mean(np.absolute(predictions - y_test))
        errors = predictions - y_test
        scoring_errors = []
        false_positives, positives, true_positives = 0, 0, 0
        false_negatives, negatives, true_negatives = 0, 0, 0
        for error in errors:
            if error < 0: # early prediction
                scoring_errors.append(math.exp((math.fabs(error))/13) - 1)

            else: # late prediction
                scoring_errors.append(math.exp((math.fabs(error)) /10) - 1)
            if error > 0:
                if error > 13:
                    false_positives += 1
                    positives += 1
                if error < 13:
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
        rmse = np.sqrt(np.mean(np.power(predictions - y_test, 2)))
        print( i  + 10, 'fold: ')
        print('MAE', model_name, mae, "Std", np.std(errors))
        print('Scoring error', model_name, scoring_error)
        print('RMSE error', model_name, rmse)
        tpr = true_positives / positives
        tnr = true_negatives / negatives
        TPRdict[model_name].append(tpr)
        TNRdict[model_name].append(tnr)
        RMSEdict[model_name].append(rmse)
        MAEdict[model_name].append(mae)
        Score08dict[model_name].append(scoring_error)
        print('Rates',model_name, tpr, tnr, '\n')

model_names_pretty = ['Raw', 'Med', 'MA',  'SG', 'DSAE', 'DGAE','DSGAN', 'DGGAN']
for model_name, model_name_pretty in zip(['_RAW', '_Med', '_MA',  '_SG', '_AE', '_AEGlob','_GAN', '_GANGlob'], model_names_pretty):
    print("&", model_name_pretty, "&",end="")
    errors = []
    for errordict, error_name in zip([MAEdict, RMSEdict, TPRdict, TNRdict, Score08dict], ['MAE', 'RMSE', 'TPR', 'TNR', 'Score08']):
        errors.append(np.round(np.mean(errordict[model_name]), 2))
    for errordict, error_name in zip([MAEdict, RMSEdict, TPRdict, TNRdict, Score08dict], ['MAE', 'RMSE', 'TPR', 'TNR', 'Score08']):
        from decimal import Decimal

        if np.round(np.mean(errordict[model_name]),2) == np.max(errors):
            print("\\textbf{", np.round(np.mean(errordict[model_name]),2), '} &', end="")# '$\pm$', np.round(np.std(errordict[model_name]),2), '&')
        else:
            if model_name == '_GANGlob':
                print(np.round(np.mean(errordict[model_name]), 2), end="")
            else:
                print(np.round(np.mean(errordict[model_name]), 2), '&', end="")
    print('\\\\')

for model_name in ['_RAW', '_Med', '_MA',  '_SG', '_AE', '_AEGlob','_GAN', '_GANGlob']:
    print('&', model_name, '&', end='')
    all_errors = []

    for errordict, error_name in zip([MAEdict, RMSEdict, TPRdict, TNRdict, Score08dict], ['MAE', 'RMSE', 'TPR', 'TNR', 'Score08']):
        end_mark = '&'
        error = np.round(np.mean(errordict[model_name]),2)
        if error_name == 'Score08':
            error = '%.2E' % Decimal(error)
            print(error, end='')
        else:
            error = "%.2f" % np.round(np.mean(errordict[model_name]),2)
            stdev_error = "%.2f" % np.round(np.std(errordict[model_name]),2)
            print(error, '$\pm$', stdev_error, end_mark, end='')
    print('\\\\')

for errordict,error_name in zip([MAEdict, RMSEdict, TPRdict, TNRdict, Score08dict], ['MAE', 'RMSE', 'TPR', 'TNR', 'Score08']):
    print(error_name)
    for model_name in ['_RAW', '_Med', '_MA',  '_SG', '_AE',  '_AEGlob','_GAN', '_GANGlob']:
        print(np.round(np.mean(errordict[model_name]),2), '&')# '$\pm$', np.round(np.std(errordict[model_name]),2), '&')
    print('\\\\')