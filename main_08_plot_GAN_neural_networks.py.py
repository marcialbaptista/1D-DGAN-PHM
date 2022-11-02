# -*- coding: utf-8 -*-
"""
main_08_plot_GAN_neural_networks.py

Plot the neural network architectures (generator and discriminator)
Only works for python < 3.6

author: Marcia Baptista (git: marcialbaptista)
"""

# Create your first MLP in Keras
import tensorflow as tf
from config import ConfigCGAN as config
from draw_nn.pptx_util import save_model_to_pptx
from draw_nn.keras_util import convert_drawer_model
from data_processing import padding_power_2

f = config.base_number_of_filters
k = config.kernel_size
s = config.strides
sz = config.train_size
c = config.channels
pad = padding_power_2((sz, sz))

inputs = tf.keras.layers.Input((sz, sz, c), name="ginput")
# inputs_pad = tf.keras.layers.ZeroPadding2D(pad, name="gpad")(inputs)

# Encoder layers
ge1 = tf.keras.layers.Conv2D(f, k, s, padding="same", name="geconv1")(inputs)

# ge2 = tf.keras.layers.LeakyReLU(config.leak, name="geact1")(ge1)
ge2 = tf.keras.layers.Conv2D(2 * f, k, s, padding="same", name="geconv2")(ge1)
# ge2 = tf.keras.layers.BatchNormalization(name="gebn2")(ge2)

# ge3 = tf.keras.layers.LeakyReLU(config.leak, name="geact2")(ge2)
ge3 = tf.keras.layers.Conv2D(4 * f, k, s, padding="same", name="geconv3")(ge2)
# ge3 = tf.keras.layers.BatchNormalization(name="gebn3")(ge3)

# ge4 = tf.keras.layers.LeakyReLU(config.leak, name="geact3")(ge3)
ge4 = tf.keras.layers.Conv2D(8 * f, k, s, padding="same", name="geconv4")(ge3)
# ge4 = tf.keras.layers.BatchNormalization(name="gebn4")(ge4)

# Decoder layers with skip connections
# gd1 = tf.keras.layers.LeakyReLU(0.0, name="geact4")(ge4)
# TODO not sure if dimensions need specifying
gd1 = tf.keras.layers.Conv2DTranspose(4 * f, k, s, padding="same", name="gdconv1")(ge4)
# gd1 = tf.keras.layers.BatchNormalization(name="gdbn1")(gd1)
# gd1 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop1")(gd1)
# gd1 = tf.keras.layers.concatenate([gd1, ge3], axis=3, name="gdcat1")

# gd2 = tf.keras.layers.LeakyReLU(0.0, name="gdact1")(gd1)
gd2 = tf.keras.layers.Conv2DTranspose(2 * f, k, s, padding="same", name="gdconv2")(gd1)
# gd2 = tf.keras.layers.BatchNormalization(name="gdbn2")(gd2)
# gd2 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop2")(gd2)
# gd2 = tf.keras.layers.concatenate([gd2, ge2], axis=3, name="gdcat2")

# gd3 = tf.keras.layers.LeakyReLU(0.0, name="gdact2")(gd2)
gd3 = tf.keras.layers.Conv2DTranspose(f, k, s, padding="same", name="gdconv3")(gd2)
# gd3 = tf.keras.layers.BatchNormalization(name="gdbn3")(gd3)
# gd3 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop3")(gd3)
# gd3 = tf.keras.layers.concatenate([gd3, ge1], axis=3, name="gdcat3")

# gd4 = tf.keras.layers.LeakyReLU(0.0)(ge1)
gd4 = tf.keras.layers.Conv2DTranspose(c, k, s, padding="same", activation="linear",
                                      name="gdconvout")(gd3)

# outputs = tf.keras.layers.Cropping2D(pad, name="gcrop")(gd4)

generator = tf.keras.models.Model(inputs=inputs, outputs=gd4, name="cond_gen")
generator2 = tf.keras.Sequential(layers=generator.layers)
for i in range(len(generator.layers)):
    print("Layer ", i, generator.layers[i])
    generator2.add(generator.layers[i])

model = convert_drawer_model(generator2)
model.save_fig("./drawings/generator.svg")

# save as pptx file
# save_model_to_pptx(model, "./drawings/generator.pptx")

inputs = tf.keras.layers.Input((sz, sz, c), name="dinput")

d0 = tf.keras.layers.Conv2D(f, k, s, padding="same", name="dconv0")(inputs)
# d0 = tf.keras.layers.LeakyReLU(config.leak, name="dact0")(d0)

d1 = tf.keras.layers.Conv2D(2 * f, k, s, padding="same", name="dconv1")(d0)
# d1 = tf.keras.layers.BatchNormalization(name="dbn1")(d1)
# d1 = tf.keras.layers.LeakyReLU(config.leak, name="dact1")(d1)

d2 = tf.keras.layers.Conv2D(4 * f, k, s, padding="same", name="dconv2")(d1)
# d2 = tf.keras.layers.BatchNormalization(name="dbn2")(d2)
# d2 = tf.keras.layers.LeakyReLU(config.leak, name="dact2")(d2)

d3 = tf.keras.layers.Conv2D(8 * f, k, s, padding="same", name="dconv3")(d2)
# d3 = tf.keras.layers.BatchNormalization(name="dbn3")(d3)
# d3 = tf.keras.layers.LeakyReLU(config.leak, name="dact3")(d3)

d4 = tf.keras.layers.Flatten(name="dflatout")(d3)

outputs = tf.keras.layers.Dense(1, name="ddenseout")(d4)

discriminator = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="cond_dsc")

discriminator2 = tf.keras.Sequential(layers=discriminator.layers)

model = convert_drawer_model(discriminator2)
model.save_fig("./drawings/discriminator.svg")

# save as pptx file
#save_model_to_pptx(model, "./drawings/discriminator.pptx")
