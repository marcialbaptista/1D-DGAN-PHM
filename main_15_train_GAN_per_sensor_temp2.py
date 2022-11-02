# -*- coding: utf-8 -*-
"""
main_09_train_GAN.py

Train the denoising GAN based on files
 - ./training_data/GAN_noisysignals.npy
 - ./training_data/GAN_puresignals.npy

author: Marcia Baptista (git: marcialbaptista)
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import utils
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as scipy_signal
import time
import os
from config import ConfigCGAN as config
import cgan as model
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
import data_processing
from collections import defaultdict
#tf.enable_eager_execution()

##################################################
# Debug parameters (show plots or not)
##################################################

show_training_data = False
generate_images = True
save_checkpoints = True

sensor_names = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']

for sensor_name_model in sensor_names:
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


    def generator_d_loss(generated_output):
        # [1,1,...,1] with generated images since we want the discriminator to judge them as real
        return tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


    def generator_abs_loss(labels, generated_images):
        # As well as "fooling" the discriminator, we want particular pressure on ground-truth accuracy
        return config.L1_lambda * tf.compat.v1.losses.absolute_difference(labels, generated_images)  # mean


    def discriminator_loss(real_output, generated_output):
        # [1,1,...,1] with real output since we want our generated examples to look like it
        real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_output), logits=real_output)

        # [0,0,...,0] with generated images since they are fake
        generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

        total_loss = real_loss + generated_loss
        return total_loss


    def train_step(inputs, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(inputs, training=True)

            real_output = discriminator(labels, training=True)
            generated_output = discriminator(generated_images, training=True)

            gen_d_loss = generator_d_loss(generated_output)
            gen_abs_loss = generator_abs_loss(labels, generated_images)
            gen_loss = gen_d_loss + gen_abs_loss
            gen_rmse = data_processing.rmse(labels, generated_images)
            gen_psnr = data_processing.psnr(labels, generated_images)
            disc_loss = discriminator_loss(real_output, generated_output)

            # Logging
            global_step.assign_add(1)
            epoch = global_step
            log_metric(gen_d_loss, "train/loss/generator_deception", epoch)
            log_metric(gen_abs_loss, "train/loss/generator_abs_error", epoch)
            log_metric(gen_loss, "train/loss/generator", epoch)
            log_metric(disc_loss, "train/loss/discriminator", epoch)
            log_metric(gen_rmse, "train/accuracy/rmse", epoch)
            log_metric(gen_psnr, "train/accuracy/psnr", epoch)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


    def log_metric(value, name, epoch):
        #with tf.compat.v1.summary.always_record_summaries():
        tf.summary.scalar(name, value, epoch)

    def train(dataset, epochs):
        global test_inputs, test_labels
        for epoch in range(epochs):
            start = time.time()

            for x, y in dataset:
                train_step(x, y)

            if epoch % config.save_per_epoch == 0:
                generate_and_save_images(generator, epoch, test_inputs[:5], test_labels[:5])

            # saving (checkpoint) the model every few epochs
            if epoch % config.save_per_epoch == 0 and save_checkpoints:
                checkpoint.save(file_prefix=checkpoint_prefix)

            #rmse, stdev = calc_RMSE(generator, selected_inputs, selected_labels)

            print('Time taken for epoch {} is {} sec'.format(epoch + 1, round(time.time() - start, 2)))


    def spline_signal(signal_1D, new_length):
        old_indices = np.arange(0, len(signal_1D))
        new_indices = np.linspace(0, len(signal_1D) - 1, new_length)
        spl = UnivariateSpline(old_indices, signal_1D, k=3, s=0)
        new_signal_1D = spl(new_indices)
        return new_signal_1D


    def calc_RMSE(model, test_inputs, test_labels):
        if model is None:
            predictions = test_inputs
        else:
            # Make sure the training parameter is set to False because we
            # don't want to train the batchnorm layer when doing inference.
            predictions = model(test_inputs, training=False)

        errors = []
        for i in range(len(predictions)):
            reconstruction = np.array(predictions[i]).reshape((config.width * config.height,))
            real = np.array(test_labels[i]).reshape((config.width * config.height,))
            error = abs(reconstruction - real)
            errors.append(error)
        return np.mean(errors), np.std(errors)


    def generate_and_save_images(model, epoch, test_inputs, test_labels):
        if model is None:
            predictions = test_inputs
        else:
            # Make sure the training parameter is set to False because we
            # don't want to train the batchnorm layer when doing inference.
            predictions = model(test_inputs, training=False)

        for i in range(5):
            reconstruction = np.array(predictions[i]).reshape((config.height * config.width,))
            noisy = np.array(test_inputs[i]).reshape((config.height * config.width,))
            real = np.array(test_labels[i]).reshape((config.height * config.width,))
            plt.plot(real, c="green", lw=3, label="Pure signal")
            plt.scatter(range(len(noisy)), noisy, c="blue", label="Synthetic noisy signal")
            plt.plot(reconstruction, c="red", lw="3", label="Reconstruction")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, 'image_at_epoch_{:04d}_{:04d}_a.png'.format(epoch, i)))
            plt.close()

        # C-MAPSS evaluation
        for i in range(5):
            unit = i + 1
            increasing_trend = increasing_trend_dic[sensor_name]
            noisy_orig = np.array(df.loc[df['unit_number'] == unit, sensor_name].values)
            all_noisy = np.array(df.loc[:, sensor_name].values)
            noisy_orig = (noisy_orig - np.mean(all_noisy)) / (np.std(all_noisy))
            if increasing_trend:
                noisy = (noisy_orig - np.min(noisy_orig)) / (np.max(noisy_orig) - np.min(noisy_orig))
            else:
                noisy = (-noisy_orig - np.min(-noisy_orig)) / (np.max(-noisy_orig) - np.min(-noisy_orig))

            noisy_signal = noisy
            # enlarge/compress the signal to height x width points
            noisy_splined = spline_signal(noisy_signal, config.height * config.width)

            # prepare the data for the convolutional format
            noisy_samples = []
            noisy_sample = (noisy_splined.reshape(config.width, config.height))
            noisy_samples.append(noisy_sample)
            noisy_samples = np.array(noisy_samples).astype('float32')
            noisy_input = noisy_samples.reshape(
                (noisy_samples.shape[0], noisy_samples.shape[1], noisy_samples.shape[2], 1))

            # denoise the data with the generative adversarial network (GAN)
            predictions = model(noisy_input, training=False)

            splined_reconstruction = np.array(predictions[0]).reshape((config.width * config.height,))
            reconstruction = scipy_signal.decimate(splined_reconstruction, 2)
            gan_signal = spline_signal(reconstruction, len(noisy))
            if increasing_trend:
                gan_signal = gan_signal * (np.max(noisy_orig) - np.min(noisy_orig)) + (np.min(noisy_orig))
            else:
                gan_signal = -(gan_signal * (np.max(-noisy_orig) - np.min(-noisy_orig)) + (np.min(-noisy_orig)))

            plt.scatter(range(len(noisy_orig)), noisy_orig, c="blue", label="C-MAPSS noisy signal (splined)")
            plt.plot(gan_signal, c="red", lw="3", label="Denoised signal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, 'CMAPSS_image_at_epoch_{:04d}_{:04d}.png'.format(epoch, i)))
            plt.close()


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

    increasing_trend_dic = defaultdict(bool)
    for sensor_name in sensor_names:
        original_signal = np.array(df.loc[df["unit_number"] == 1, sensor_name].values)
        if original_signal[-1] >= original_signal[0]:
            increasing_trend_dic[sensor_name] = True
        else:
            increasing_trend_dic[sensor_name] = False


    # Load data
    y_noisy = np.load('./training_data/' + sensor_name_model + '/GAN_noisy_signals.npy', allow_pickle=True)
    y_pure = np.load('./training_data/' + sensor_name_model + '/GAN_pure_signals.npy', allow_pickle=True)
    print("Number of total training units:", int(len(y_pure)))

    import itertools

    y_noisy_sensor_reshaped = list(itertools.chain.from_iterable(y_noisy))
    min_value = np.min(y_noisy_sensor_reshaped)
    max_value = np.max(y_noisy_sensor_reshaped)

    # Reshape data
    y_noisy_reshaped = []
    y_pure_reshaped = []
    for i in range(len(y_noisy)):
        noisy_sample = (y_noisy[i])
        pure_sample = (y_pure[i])

        #noisy_sample = (noisy_sample - min_value) / (max_value - min_value)
        #pure_sample = (pure_sample - min_value) / (max_value - min_value)

        if show_training_data:
            plt.plot(noisy_sample)
            plt.plot(pure_sample)
            plt.show()
        pure_sample = spline_signal(pure_sample, config.width * config.height)
        noisy_sample = spline_signal(noisy_sample, config.width * config.height)
        noisy_sample = noisy_sample.reshape(config.width, config.height)
        pure_sample = pure_sample.reshape(config.width, config.height)
        y_noisy_reshaped.append(noisy_sample)
        y_pure_reshaped.append(pure_sample)
    y_noisy_reshaped = np.array(y_noisy_reshaped)
    y_pure_reshaped = np.array(y_pure_reshaped)
    noisy_input = y_noisy_reshaped.reshape((int(y_noisy_reshaped.shape[0]), y_noisy_reshaped.shape[1], y_noisy_reshaped.shape[2], 1))
    pure_input = y_pure_reshaped.reshape((y_pure_reshaped.shape[0], y_pure_reshaped.shape[1], y_pure_reshaped.shape[2], 1))

    model_path = os.path.join('models_GAN', sensor_name_model, 'model')
    results_path = os.path.join('models_GAN', sensor_name_model, 'results')

    # Make directories for this run
    utils.safe_makedirs(model_path)
    utils.safe_makedirs(results_path)

    # # Initialise logging
    log_path = os.path.join('logs', sensor_name_model)
    summary_writer = tf.summary.create_file_writer(log_path)
    summary_writer.set_as_default()
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Load the dataset
    M = int(len(pure_input)*0.8)
    train_images = pure_input[:M]
    noisy_train_images = noisy_input[:M]
    test_images = pure_input[M:]
    noisy_test_images = noisy_input[M:]

    train_labels = train_images.reshape(train_images.shape[0],
                                        config.raw_size,
                                        config.raw_size,
                                        config.channels).astype('float32')

    train_inputs = noisy_train_images.reshape(noisy_train_images.shape[0],
                                              config.raw_size,
                                              config.raw_size,
                                              config.channels).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)) \
        .shuffle(config.buffer_size).batch(config.batch_size)

    # Test set
    test_labels = test_images.reshape(test_images.shape[0],
                                      config.raw_size,
                                      config.raw_size,
                                      config.channels).astype('float32')

    test_inputs = noisy_test_images.reshape(noisy_test_images.shape[0],
                                              config.raw_size,
                                              config.raw_size,
                                              config.channels).astype('float32')

    # Set up the models for training
    generator = model.make_generator_model_small()
    discriminator = model.make_discriminator_model()

    generator_optimizer = tf.optimizers.Adam(config.learning_rate)
    discriminator_optimizer = tf.optimizers.Adam(config.learning_rate)

    checkpoint_prefix = os.path.join(model_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    print("\nTraining...\n")
    # Compile training function into a callable TensorFlow graph (speeds up execution)
    train_step = tf.function(train_step)

    no_epochs = 1000
    train(train_dataset, no_epochs)
    print("\nTraining done\n")

# View training logs as follows in the terminal
# tensorboard --logdir=C:\Users\Owner\Downloads\denoising_gan\logs\global\final