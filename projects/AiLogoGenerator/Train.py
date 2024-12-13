import DataLoader
from numpy import np
import matplotlib.pyplot as plt

from ModelGan import GANModel


def train_gan(generator, discriminator, gan, data, latent_dim, epochs=5000, batch_size=64, save_interval=500):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_imgs = data[idx]

        z = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(z)

        real_loss = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        fake_loss = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(real_loss, fake_loss)

        # Train Generator
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(z, valid_y)

        # Display progress
        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            save_generated_images(generator, epoch, latent_dim)


def save_generated_images(generator, epoch, latent_dim, examples=10):
    z = np.random.normal(0, 1, (examples, latent_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    for i, img in enumerate(gen_imgs):
        plt.subplot(1, examples, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"generated_logo_{epoch}.png")
    plt.close()

# Example Usage
latent_dim = 100
img_shape = (64, 64, 3)

gan_model = GANModel(latent_dim, img_shape)
data_loader = DataLoader

gan_model.summary()
generator = gan_model.build_generator()
discriminator = gan_model.build_discriminator()
logos = data_loader.load_images()
train_gan(generator, discriminator, gan_model, logos, latent_dim)
