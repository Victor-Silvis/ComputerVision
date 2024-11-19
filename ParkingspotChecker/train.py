'''File for training the model'''

'''
NOTE this is a relatively simple implementation. No hypertuning is used to optimize the model.
Furthermore, I expect futher improvement can be gained with more image preprocessing. However,
for this usecase (parkingspot) the model is already more than adequate.

'''

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,MaxPooling2D,Flatten,Conv2D
from tensorflow.keras import optimizers


MODEL_SAVENAME = 'parkingspotModel.keras'
DATA_DIR = 'train_data'
EMPTY_DIR = os.path.join(DATA_DIR, 'empty')
NOT_EMPTY_DIR = os.path.join(DATA_DIR, 'not_empty')
IMG_WIDTH = 15
IMG_HEIGHT = 15

#Load images, process and label them
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        images.append(img_array)
        labels.append(label)
    
    return np.array(images), np.array(labels)

empty_imgs, empty_labels = load_images(EMPTY_DIR, 0)
not_empty_imgs, not_empty_labels = load_images(NOT_EMPTY_DIR, 1)
images = np.concatenate((not_empty_imgs, empty_imgs), axis=0)
labels = np.concatenate((not_empty_labels, empty_labels), axis=0)
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

#Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Define a faster and simpler model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(15, 15, 3), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-3),
              metrics=['accuracy'])

#Training of model
batch_size = 32
epochs = 14
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True)

#Save Model
model.save(MODEL_SAVENAME)
print('Model Trained and Saved')