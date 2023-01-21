import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class Autoencoder(Model):

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 1)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'), 
            layers.Dense(526, activation='relu'), 
            layers.Dense(1024, activation='relu'),
            layers.Reshape((32, 32))])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded