'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

#setup

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import metrics
import functools

# dimensions of our images.
img_width, img_height = 140, 140

class_number=4


train_data_dir = 'navigation_dataset/train'
validation_data_dir = 'navigation_dataset/validation'
nb_train_samples = 4200
nb_validation_samples = 1450

epochs = 20
batch_size = 50


#%% 
#network setup

if K.image_data_format() == 'channels_first':
    input_shape = Input(shape=(3, img_width, img_height))
else:
    input_shape = Input(shape=(img_width, img_height, 3))

def droneNet(inputs=None, include_top=True, classes=10, *args, **kwargs):
    if inputs is None :
        if K.image_data_format() == 'channels_first':
            input_shape = Input(shape=(3, 224, 224))
        else:
            input_shape = Input(shape=(224, 224, 3))
    else:
        input_shape=inputs

    outputs = []

    x = Conv2D(32, (3, 3), strides=(1, 1),use_bias=False)(input_shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    outputs.append(x)

    for i in range(3):
        x = Conv2D(64*(2**i), (3, 3), strides=(1, 1),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        outputs.append(x)

    x = Conv2D(256, (3, 3), strides=(1, 1),use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='sigmoid')(x)
        return Model(inputs=input_shape, outputs=x, *args, **kwargs)
    else:
        return Model(inputs=input_shape, outputs=outputs, *args, **kwargs)
    
model=droneNet(inputs=input_shape,classes=class_number)
model.summary()

def top2_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top2_acc])


#%%
#data preprocessing

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#%% 
#train

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)
    

model.save_weights('navigation_droneNet_v2_140x140_weight.h5')
model.save('navigation_droneNet_v2_140x140_model.h5')

#%%
#evaluate
    
#building confusion matrix
import numpy as np

evaluation_datagen = ImageDataGenerator(rescale=1. / 255)
evaluation_generator = evaluation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_result = model.evaluate_generator(evaluation_generator, steps=nb_validation_samples//batch_size)

prediction = model.predict_generator(evaluation_generator,nb_validation_samples//batch_size)
prediction = np.argmax(prediction, axis=1)
        
import pandas as pd
pd.crosstab(evaluation_generator.classes,prediction,
            rownames=['label'],colnames=['predict'])
print(evaluation_generator.class_indices)
