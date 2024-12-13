import tensorflow as tf
from keras import Model
from tensorflow.keras import layers

class GANModel:
    def __init__(self, latent_dim, img_shape):
        """
        Initialize the GANModel class with generator and discriminator.
        :param latent_dim: Dimension of the latent space (input noise vector)
        :param img_shape: Shape of the image (e.g., (64, 64, 3))
        """
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self._compile_gan()

    def build_discriminator(self):
        """Build the discriminator model."""
        model = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=self.img_shape),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def build_generator(self):
        """Build the generator model."""
        model = tf.keras.Sequential([
            layers.Dense(8 * 8 * 256),
            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh")
        ])
        return model

    def _compile_gan(self):
        """Combine generator and discriminator into a GAN model."""
        self.discriminator.trainable = False  # Freeze discriminator during generator training

        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        gan = Model(z, valid)
        gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
        return gan

    def summary(self):
        """Print summaries of the generator, discriminator, and GAN."""
        print("Generator Model:")
        self.generator.summary()
        print("\nDiscriminator Model:")
        self.discriminator.summary()
        print("\nGAN Model:")
        self.gan.summary()

