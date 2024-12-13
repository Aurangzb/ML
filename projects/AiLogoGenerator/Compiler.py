# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

# GAN model
discriminator.trainable = False
z = layers.Input(shape=(latent_dim,))
img = generator(z)
valid = discriminator(img)

gan = tf.keras.Model(z, valid)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
