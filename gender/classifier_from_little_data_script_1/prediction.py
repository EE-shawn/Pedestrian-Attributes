# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:12:45 2017

@author: TWang

after traning model, make prediction 

"""

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf

import cv2

# dimensions of our images.
img_width, img_height = 100, 250

input_shape = (img_width, img_height, 3)

#==============================================================================
# test_model = Sequential()
# 
# test_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# test_model.add(Activation('relu'))
# test_model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# test_model.add(Conv2D(32, (3, 3)))
# test_model.add(Activation('relu'))
# test_model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# test_model.add(Conv2D(64, (3, 3)))
# test_model.add(Activation('relu'))
# test_model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# test_model.add(Flatten())
# test_model.add(Dense(64))
# test_model.add(Activation('relu'))
# test_model.add(Dropout(0.5))
# test_model.add(Dense(1))
# test_model.add(Activation('sigmoid'))
#==============================================================================

test_model = load_model('first_model.h5')

def predict(basedir, model):
    
    for i in range(1,30):
        
        image_path = basedir + str(i) + '.jpg'
        
        orig = cv2.imread(image_path)

        print("[INFO] loading and preprocessing image...")
        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)
        
        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(image)
    
        probabilities = model.predict_proba(image)
        
        inID = class_predicted[0]
        
        if inID == 0 :
            label = 'female'
        else :
            label = 'male'
            
        # get the prediction label
        print("Image ID: {}, Label: {}, Probability:{}".format(inID, label, probabilities))
        
        # display the predictions with the image
        cv2.putText(orig, "{}".format(label), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (43, 99, 255), 2)
    
        cv2.imshow("Classification", orig)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
#==============================================================================
# basedir = r"D:\TF_Try\gender\data\test\female\female"
# predict(basedir, test_model)
#==============================================================================

basedir = r"D:\TF_Try\gender\data\test\office_pedestrian\person"
predict(basedir, test_model)

#basedir = r"D:\TF_Try\gender\data\test\male\male"
#predict(basedir, test_model)



print('done')
