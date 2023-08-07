import tensorflow as tf
import numpy as np

# Create synthetic data for the simulation (you can replace this with your real data)
def generate_real_data(num_samples):
    return np.random.normal(5, 1, (num_samples, 1))

# Create the generator model
def create_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(1,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model

# Create the discriminator model
def create_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(1,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create the GAN model
def create_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Training the GAN
def train_gan(gan, real_data, num_epochs, batch_size):
    for epoch in range(num_epochs):
        for _ in range(real_data.shape[0] // batch_size):
            noise = np.random.normal(0, 1, size=(batch_size, 1))
            fake_data = gan.layers[0].predict(noise)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = gan.layers[1].train_on_batch(real_data, real_labels)
            d_loss_fake = gan.layers[1].train_on_batch(fake_data, fake_labels)

            # Train the generator
            noise = np.random.normal(0, 1, size=(batch_size, 1))
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

        print(f"Epoch: {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# Main function
if __name__ == "__main__":
    # Generate real data
    num_samples = 1000
    real_data = generate_real_data(num_samples)

    # Create models
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)

    # Compile models
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    # Train the GAN
    num_epochs = 10000
    batch_size = 32
    train_gan(gan, real_data, num_epochs, batch_size)

    # Generate synthetic data
    noise = np.random.normal(0, 1, size=(num_samples, 1))
    synthetic_data = generator.predict(noise)

    print("Synthetic Data:", synthetic_data)
