import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt

# Ensure TensorFlow is using CPU
tf.config.set_visible_devices([], 'GPU')

# Helper function to load and preprocess images
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [256, 256])
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

# Generator definition (same as the Pix2Pix generator)
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  
        downsample(128, 4),  
        downsample(256, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  
        upsample(512, 4, apply_dropout=True),  
        upsample(512, 4, apply_dropout=True),  
        upsample(512, 4),  
        upsample(256, 4),  
        upsample(128, 4),  
        upsample(64, 4),  
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Loss function for generator
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  # L1 loss for reconstruction
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss

# Function to train for a single step
@tf.function
def train_step(input_image, target_image, generator, generator_optimizer):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)

        # Normally, we would use a discriminator here. 
        # For simplicity, we'll skip it and just optimize the generator loss.
        disc_generated_output = tf.ones_like(gen_output)  # Simulate perfect discriminator
        gen_loss = generator_loss(disc_generated_output, gen_output, target_image)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_output, gen_loss


# Main function to load images, train, and output a result
def main():
    # Load the input and target images
    input_image = load_image('input_image.jpg')
    target_image = load_image('target_image.jpg')

    # Print image shape to verify that the images are loaded properly
    print(f'Input Image Shape: {input_image.shape}')
    print(f'Target Image Shape: {target_image.shape}')

    # Add a batch dimension (since the model expects a batch of images)
    input_image = tf.expand_dims(input_image, axis=0)
    target_image = tf.expand_dims(target_image, axis=0)

    # Create the generator model and optimizer
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Number of training iterations (set to 100 for better output)
    num_iterations = 100

    # Train for multiple iterations
    for i in range(num_iterations):
        gen_output, gen_loss = train_step(input_image, target_image, generator, generator_optimizer)

        # Print the generator loss every few iterations
        if (i+1) % 10 == 0:
            print(f'Iteration {i+1}, Generator Loss: {gen_loss.numpy()}')

        # Plot intermediate results after every 20 iterations
        if (i+1) % 20 == 0:
            # De-normalize the generated output for visualization
            gen_output_de_normalized = (gen_output[0] + 1) / 2  # Convert back to [0, 1] range
            plt.imshow(gen_output_de_normalized)
            plt.title(f'Generated Image after {i+1} iterations')
            plt.axis('off')
            plt.show()

    # After training, show final results
    # Remove the batch dimension and de-normalize the generated output
    final_output = (gen_output[0] + 1) / 2  # Convert back to [0, 1] range

    # Plot the input, target, and final generated image
    plt.figure(figsize=(15, 5))

    display_list = [input_image[0], target_image[0], final_output]
    title = ['Input Image', 'Target Image', 'Generated Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Re-scale for display
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
