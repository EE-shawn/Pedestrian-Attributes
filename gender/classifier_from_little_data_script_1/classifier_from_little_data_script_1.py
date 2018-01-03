# -*- coding: utf-8 -*-
'''
Created on Thu Dec 21 15:47:25 2017

@author: TWang

"Building powerful image classification models using very little data"

It uses data that can be downloaded at:  https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats

- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.

'''
import os
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import math


# dimensions of our images.
img_width, img_height = 100, 250

train_data_dir = r'D:\WANG Tao\gender\data\train'
validation_data_dir = r'D:\WANG Tao\gender\data\validation'

nb_train_samples = 90000
nb_validation_samples = 10000
epochs = 30
batch_size = 200

'''Tensforflow and Theno has different img channel order'''
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
'''
step1 :
end the model with a single unit and a sigmoid activation, which is perfect for a binary classification. 
To go with it we will also use the binary_crossentropy loss to train our model.'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


'''
step2:
Let's prepare data. 
We will use .flow_from_directory() to generate batches of image data 
(and their labels) directly from our jpgs in their respective folders.
'''
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    train_data_dir, # this is the target directory
    target_size=(img_width, img_height), # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary') # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir, 
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

savedir = r'D:\WANG Tao\gender\classifier_from_little_data_script_1'
logdir = os.path.join(savedir,'logs')
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
tensorboard = TensorBoard(log_dir=logdir,batch_size=batch_size)

'''
step3:
We can now use these generators to train our model. 
Each epoch takes 20-30s on GPU and 300-400s on CPU. 
So it's definitely viable to run this model on CPU if you aren't in a hurry.
'''
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_model_weights.h5') # always save your weights after training or during training
model.save('first_model.h5')


'''figure'''
plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

'''
result:
    This approach gets us to a validation accuracy of 0.79-0.81 after 50 epochs (a number that was picked arbitrarily 
    --because the model is small and uses aggressive dropout, it does not seem to be overfitting too much by that point). 
    So at the time the Kaggle competition was launched, we would be already be "state of the art" 
    --with 8% of the data, and no effort to optimize our architecture or hyperparameters. 
    In fact, in the Kaggle competition, this model would have scored in the top 100 (out of 215 entrants). 
    I guess that at least 115 entrants weren't using deep learning ;)

    Note that the variance of the validation accuracy is fairly high, both because accuracy is a high-variance metric and because we only use 800 validation samples. 
    A good validation strategy in such cases would be to do k-fold cross-validation, but this would require training k models for every evaluation round.
'''