# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:31:37 2018

@author: Yung-Yu Tsai

evaluate accuracy of model weight
"""


#setup

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import metrics
from quantize_simulator.utils_tool.confusion_matrix import show_confusion_matrix
import time
import numpy as np

# dimensions of our images.
img_width, img_height = 250, 140

class_number=4


train_data_dir = '../navigation_dataset/train'
validation_data_dir = '../navigation_dataset/validation'
nb_train_samples = 4200
nb_validation_samples = 1450

epochs = 20
batch_size = 50

#%%
# model setup

def top2_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

model=load_model('../navigation_droneNet_v1_250x140_model.h5',custom_objects={'top2_acc':top2_acc})
model.load_weights('../navigation_droneNet_v1_250x140_weight.h5')

model.summary()

#%%
# evaluate model

evaluation_datagen = ImageDataGenerator(rescale=1. / 255)
evaluation_generator = evaluation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

t = time.time()

test_result = model.evaluate_generator(evaluation_generator, steps=nb_validation_samples//batch_size)

t = time.time()-t

prediction = model.predict_generator(evaluation_generator,nb_validation_samples//batch_size)
prediction = np.argmax(prediction, axis=1)

print('\nruntime: %f s'%t)        
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

show_confusion_matrix(evaluation_generator.classes,prediction,evaluation_generator.class_indices.keys(),'Confusion Matrix',normalize=False)

