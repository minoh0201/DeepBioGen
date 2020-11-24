# Fix numpy random seed
from numpy.random import seed
seed(0)
# Fix tensorflow random seed
from tensorflow import set_random_seed
set_random_seed(0)

import time

import tensorflow as tf

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, Cropping2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from functools import partial

import keras.backend as K

from keras.backend.tensorflow_backend import set_session
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import matplotlib.pyplot as plt

import math

import numpy as np
from numpy.random import randint
from numpy.random import randn

from experimentor import Experimentor

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

batch_size_global = None

#############
# Generate column-wise clustered samples using conditional Wasserstein GAN with gradient penalty
def deepbiogen(exp : Experimentor, aug_rates : list, num_clusters : int=1, max_gans : int=1, num_epochs : int=6000, batch_size: int=128, sample_interval=2000):
    # Time stamp
    start_time = time.time()

    # Set augmentation name
    exp.aug_name = f"deepbiogen_c{num_clusters}_gmax{max_gans}_e{num_epochs}"

    # Store aug rates
    exp.aug_rates = aug_rates

    # The largest number of augmented samples
    max_aug_samples = exp.X_train.shape[0] * aug_rates[-1]

    # Re-ordering features with clustering algorithm
    exp.X_train, order = kmeans_ordering(exp.X_train, num_clusters)
    exp.X_test = reorder_test(exp.X_test, order)

    # Fake sample holder
    pool_X_fakes = []
    pool_y_fakes = []

    # Load GANs and Generate fake samples
    for i in range(max_gans):

        # Train a single WGAN
        wgan = CWGANGP(X_train=exp.X_train, 
                        y_train=exp.y_train, 
                        model_path=exp.model_path, 
                        model_name=f"deepbiogen_c{num_clusters}_g{i+1}", 
                        epochs=num_epochs,
                        batch_size=batch_size,
                        sample_interval=sample_interval)

        # Train a single WGAN
        if not os.path.exists(os.path.join(wgan.model_path, wgan.model_name + f'_e{num_epochs}_generator')):
            wgan.train()

        # Max number of samples for each WGAN 
        n_samples_forEachGAN = int(max_aug_samples / (i+1))

        #Augmenting X_fake
        print(f'Generating {n_samples_forEachGAN} fake data points...')
        X_fake, y_fake = wgan.generate(n_samples=n_samples_forEachGAN, epoch=num_epochs)
        pool_X_fakes.append(X_fake)
        pool_y_fakes.append(y_fake)

    # Store augmentations
    exp.X_train_augs = []
    exp.y_train_augs = []
    exp.X_augs = []
    exp.y_augs = []

    for i in range(max_gans):
        # Training data + Augmentation data
        X_train_augs_by_rates = []
        y_train_augs_by_rates = []
        # Augmentation data alone
        X_augs_by_rates = []
        y_augs_by_rates = []

        for aug_rate in aug_rates:
            X_temp = exp.X_train
            y_temp = exp.y_train

            for j in range(i+1):
                num_aug_samples_for_each_gan = int(exp.X_train.shape[0] * (aug_rate / (i+1)))
                X_temp = np.concatenate((X_temp, pool_X_fakes[j][:num_aug_samples_for_each_gan]))
                y_temp = np.concatenate((y_temp, pool_y_fakes[j][:num_aug_samples_for_each_gan]))
            
            # Training data + Augmentation data
            X_train_augs_by_rates.append(X_temp)
            y_train_augs_by_rates.append(y_temp)
            # Augmentation data alone
            X_augs_by_rates.append(X_temp[exp.X_train.shape[0]:])
            y_augs_by_rates.append(y_temp[exp.X_train.shape[0]:])
        
        # Training data + Augmentation data
        exp.X_train_augs.append(X_train_augs_by_rates)
        exp.y_train_augs.append(y_train_augs_by_rates)
        # Augmentation data alone
        exp.X_augs.append(X_augs_by_rates)
        exp.y_augs.append(y_augs_by_rates)

    print(f"--- Augmented with {exp.aug_name} in {round(time.time() - start_time, 2)} seconds ---")

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        global batch_size_global
        alpha = K.random_uniform((batch_size_global, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
class CWGANGP():
    def __init__(self, X_train, y_train, model_path, model_name, epochs=100, batch_size=32, sample_interval=50):
        self.X_train = X_train
        self.X_train_reshaped = X_train.reshape((X_train.shape[0],1,X_train.shape[1],1))
        self.y_train = y_train
        self.img_rows = 1 #28
        self.img_cols = X_train.shape[1] #28
        self.channels = 1
        self.nclasses = 2 #10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.losslog = []
        self.epochs = epochs
        self.batch_size = batch_size
        global batch_size_global
        batch_size_global = batch_size
        self.sample_interval = sample_interval
        self.model_path = model_path
        self.model_name = model_name
        
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        #optimizer = RMSprop(lr=0.00005)
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        #self.generator.summary()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        
        # Generate image based of noise (fake sample) and add label to the input 
        label = Input(shape=(1,))
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, label])
        valid = self.critic([real_img, label])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()(inputs=[real_img, fake_img])
        
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, label, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # add label to the input
        label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([img, label])
        # Defines generator model
        self.generator_model = Model([z_gen, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        
        noise = Input(shape=(100,))
        label = Input(shape=(1,))
        
        li = Flatten()(Embedding(self.nclasses, 50)(label))
        li = Dense(64)(li)
        li = Reshape((1, 64, 1))(li)

        gen = Dense(1*64*256)(noise)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((1, 64, 256))(gen)

        merge = Concatenate()([gen, li])
        
        gen = Conv2DTranspose(128,(1, 32), strides=(1, 1), padding='same', use_bias=False)(merge) #gen
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2DTranspose(64,(1, 32), strides=(1, 2), padding='same', use_bias=False)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)

        gen = Conv2DTranspose(1,(1, 128), strides=(1, 2), padding='same', use_bias=False)(gen)
        
        crop = gen.shape[2] - self.img_cols
        #gen = Cropping2D( cropping=( (0,0),(0, crop) ), data_format="channels_last")(gen)

        model = Model(inputs = [noise, label], outputs = gen)
        
        return model


    def build_critic(self):
        
        # label input
        label = Input(shape=(1,))
        # embedding for categorical input
        li = Flatten()(Embedding(self.nclasses, 50)(label))
        
        # scale up to image dimensions with linear activation
        n_nodes = self.img_rows * self.img_cols
        print(li)
        print(n_nodes)
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((self.img_rows, self.img_cols, 1))(li)

        img = Input(shape=self.img_shape)
        # concat label as a channel
        merge = Concatenate()([img, li])
    
        critic = Conv2D(64, (1, 5), strides=(1, 2), padding='same')(merge) #
        critic = LeakyReLU()(critic)
        critic = Dropout(0.3)(critic)

        critic = Conv2D(128, (1, 5), strides=(1, 2), padding='same')(critic)
        critic = LeakyReLU()(critic)
        critic = Dropout(0.3)(critic)

        critic = Flatten()(critic)
        critic = Dense(1)(critic)

        model = Model(inputs = [img, label], outputs = critic)
        
        return model
        

    def train(self):
        
        print(f'X_train shape: {self.X_train.shape}')
        
        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(self.epochs + 1):
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Select a random batch of images
                idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
                imgs, labels = self.X_train_reshaped[idx], self.y_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, labels, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.nclasses, self.batch_size).reshape(-1, 1)
            g_loss = self.generator_model.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            self.losslog.append([d_loss[0], g_loss])
            # If at save interval => save generated image samples
            if epoch % self.sample_interval == 0:
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                #self.sample_images(epoch, self.X_train.shape[0])
                self.generator.save_weights(os.path.join(self.model_path, self.model_name + f'_e{epoch}_generator'), overwrite=True)
                self.critic.save_weights(os.path.join(self.model_path, self.model_name + f'_e{epoch}_discriminator'), overwrite=True)
                # with open('loss.log', 'w') as f:
                #     f.writelines('d_loss, g_loss\n')
                #     for each in self.losslog:
                #         f.writelines('%s, %s\n'%(each[0], each[1]))

    def sample_images(self, epoch, n_samples):
        noise = randn(self.latent_dim * n_samples)
        noise = noise.reshape(n_samples, self.latent_dim)
        labels = randint(0, self.nclasses, n_samples)
        
        gen_imgs = self.generator.predict([noise, labels])
        
        #self.viz(gen_imgs.reshape((-1,self.X_train.shape[1])), labels)

        #plt.savefig("images/mnist_%d.png" % epoch)
        #plt.close()

    def generate(self, n_samples, epoch):
        self.generator.load_weights(os.path.join(self.model_path, self.model_name + f'_e{epoch}_generator'))
        noise = randn(self.latent_dim * n_samples)
        noise = noise.reshape(n_samples, self.latent_dim)
        labels = randint(0, self.nclasses, n_samples)
        gen_imgs = self.generator.predict([noise, labels])
        #self.viz(gen_imgs.reshape((-1,self.X_train.shape[1])), labels)
        return gen_imgs.reshape((-1, self.X_train.shape[1])), labels
    

##############
# re-ordering
def kmeans_ordering(X, n_clusters):
    model = KMeans(n_clusters = n_clusters, random_state = 0).fit(X.T)
    order = np.argsort(model.labels_)
    #print(sorted(collections.Counter(model.labels_).items()))
    return (X.T[order]).T, order

def hierarchical_ordering(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters).fit(X.T)
    order = np.argsort(model.labels_)
    #print(sorted(collections.Counter(model.labels_).items()))
    return (X.T[order]).T, order

def reorder_test(X_test, order):
    return (X_test.T[order]).T
