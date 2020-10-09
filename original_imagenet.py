# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:31:37 2018

@author: Yung-Yu Tsai

evaluate accuracy of model weight on imagenet validation set
"""


#setup

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
import time
import numpy as np

# dimensions of our images.
img_width, img_height = 224, 224

class_number=1000

validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
nb_validation_samples = 50000

#%%
# model setup

def top5_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)

print('Building model...')
t = time.time()
#model = ResNet50(weights='../resnet50_weights_tf_dim_ordering_tf_kernels.h5')
model = MobileNet(weights='../mobilenet_1_0_224_tf.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top5_acc])

t = time.time()-t
#model.summary()

print('model build time: %f s'%t)

#%%

img_path = '../test_images/wine.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=5)[0])



#%%
# evaluate model

print('preparing dataset...')

#evaluation_datagen = ImageDataGenerator(rescale=1. / 255)
evaluation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
evaluation_generator = evaluation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    shuffle=False)

print('dataset ready')

t = time.time()
print('evaluating...')

test_result = model.evaluate(evaluation_generator, verbose=1)

t = time.time()-t
print('evaluate done')
print('\nruntime: %f s'%t)        
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top5 accuracy:', test_result[2])

#%%

prediction = model.predict(evaluation_generator, verbose=1)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(evaluation_generator.classes,prediction,evaluation_generator.class_indices.keys(),'Confusion Matrix',figsize=(10,8),normalize=False,big_matrix=True)


