# Import libraries necessary for this project.
import json
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Location of the simulator data.
DATA_FILE = './driving_data/driving_log.csv'
local_img_path = './driving_data/IMG/'

# Load the training data from the simulator.
cols = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
data = pd.read_csv(DATA_FILE, names=cols, header=1)

# Separate the image paths and steering angles.
images = data[['Center Image', 'Left Image', 'Right Image']]
angles = data['Steering Angle']

# Split the data into training and validation sets.
images_train, images_validation, angles_train, angles_validation = train_test_split(images, angles, test_size=0.15, random_state=42)

# Define the model
model = Sequential()
# cropping top and bottom of image remove trees and sky
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalization layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

# Print out summary of the model
model.summary()
# Select the optimizer and compile the model.
optimizer = Adam(lr=0.0001)
model.compile(loss='mse', optimizer='adam')

# Helper to load an normalize images.
def load_image(path, flip=False):
    # get local path to file
    filename = path.split('/')[-1]
    local_path = local_img_path + filename
    
    # Read the image from disk, and flip it if requested.
    image = cv2.imread(local_path)
    if flip:
        image = cv2.flip(image, 1)
    
    # Return the normalized image.
    return image

# Data generator.
def generate_batches(images, angles, batch_size=64, augment=True):
    # Create an array of sample indexes.
    indexes = np.arange(len(images))
    batch_images = []
    batch_angles = []
    sample_index = 0
    while True:
        # Reshuffle the indexes after each pass through the samples to minimize
        # overfitting on the data.
        np.random.shuffle(indexes)
        for i in indexes:
            # Increment the number of samples. 
            sample_index += 1
            
            # Load the center image and weight.
            center_image = load_image(images.iloc[i]['Center Image'])
            center_angle = float(angles.iloc[i])
            batch_images.append(center_image)
            batch_angles.append(center_angle)
            
            # Add augmentation if requested
            if augment:
                # Load the flipped image and invert angle
                flipped_image = load_image(images.iloc[i]['Center Image'], True)
                flipped_angle = -1. * center_angle
                batch_images.append(flipped_image)
                batch_angles.append(flipped_angle)

                # Load the left image and adjust angle
                left_image = load_image(images.iloc[i]['Left Image'])
                left_angle = min(1.0, center_angle + 0.25)
                batch_images.append(left_image)
                batch_angles.append(left_angle)
                # Load the right image and adjust angle
                right_image = load_image(images.iloc[i]['Right Image'])
                right_angle = max(-1.0, center_angle - 0.25)
                batch_images.append(right_image)
                batch_angles.append(right_angle)
            
            # If we have processed batch_size samples or this is the last batch
            # of the epoch, then submit the batch. Note that due to augmentation
            # there may be more than batch_size elements in the batch.
            if (sample_index % batch_size) == 0 or (sample_index % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_angles)
                batch_images = []
                batch_angles = []

# Instantiate data generators for training and validation.
nb_epoch = 5
samples_per_epoch = 4 * len(images_train)
generator_train = generate_batches(images_train, angles_train)
nb_val_samples = len(images_validation)
generator_validation = generate_batches(images_validation, angles_validation, augment=False)

# Generator callbacks (TensorBoard, CSVLogger)
tensorboard = TensorBoard(log_dir='./revised_model_tensorboard', histogram_freq=1, 
                          write_graph=True, write_images=True)

csv_logger = CSVLogger('revised_model_log.csv', append=True, separator=';')

callbacksList = [tensorboard, csv_logger]

# Run the model.
history = model.fit_generator(
    generator_train, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
    validation_data=generator_validation, nb_val_samples=nb_val_samples, callbacks=callbacksList)

# plot model and save to 
model.save('revised_model.h5')

# Save the generated model and weights.
# Save model as json file
json_string = model.to_json()

with open('revised_model.json', 'w') as outfile:
    json.dump(json_string, outfile)
    
    # save weights
    model.save_weights('./revised_model_weights.h5')
    print("Saved")
    

# plot model
#plt.plot(history.history(['loss']))
#plt.plot(history.history(['val_loss']))
#plt.title('Model Mean-Squared Error Loss')
#plt.ylabel('Mean-Squared Error Loss')
#plt.xlabel('Epoch')
#plt.legend(['Training Set', 'Validation Set'], loc='upper right')
#plt.savefig('nvidia__edited_plot')
