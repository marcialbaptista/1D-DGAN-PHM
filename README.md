# GAN denoiser

Tensorflow/Keras implementation of a Conditional Generative Adversarial Network (CGAN) model that can be used for image denoising or artefact removal in PHM

The CGAN consists of a generator network and a discriminator network. The generator takes noisy/artefact images as input, with the objective of getting as close to the true image as possible. The discriminator model takes true images or generated images as input, with the objective of distinguishing the two as accurately as possible. 
