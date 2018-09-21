# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:31:37 2018

@author: Yung-Yu Tsai

evaluate accuracy of model weight on imagenet validation set
"""


#setup

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras import metrics
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from quantize_simulator.utils_tool.confusion_matrix import show_confusion_matrix
import time
import numpy as np

# dimensions of our images.
img_width, img_height = 224, 224

class_number=1000

validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
nb_validation_samples = 50000

epochs = 20
batch_size = 50

#%%
# model setup

def top5_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)

print('Building model...')

model = MobileNet(weights='../mobilenet_1_0_224_tf.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top5_acc])
model.summary()

#%%

#img_path = '../test_images/cars.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=5)[0])



#%%
# evaluate model

print('preparing dataset...')

evaluation_datagen = ImageDataGenerator(rescale=1. / 255)
evaluation_generator = evaluation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

print('dataset ready')

t = time.time()
print('evaluating...')

test_result = model.evaluate_generator(evaluation_generator, steps=nb_validation_samples//batch_size)

t = time.time()-t
print('evaluate done')
print('\nruntime: %f s'%t)        
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top5 accuracy:', test_result[2])

#%%

prediction = model.predict_generator(evaluation_generator,nb_validation_samples//batch_size)
prediction = np.argmax(prediction, axis=1)

#show_confusion_matrix(evaluation_generator.classes,prediction,evaluation_generator.class_indices.keys(),'Confusion Matrix',normalize=False)


