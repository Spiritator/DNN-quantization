# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


def dataset_setup(dataset, 
                  img_rows = 224, img_cols = 224, 
                  num_classes = 10, batch_size=32, 
                  data_augmentation = False, data_dir = None, 
                  preprocessing_function=None,
                  verbose=2):
    """
    Dataset Setup Wrapper
        Dataset prepare automation for Mnist, Cifar10 or Keras ImageDataGenerator.

    Parameters
    ----------
    dataset : String. One of 'Mnsit', 'Cifar10', 'ImageDataGenerator'.
        The data set to be prepared.
    img_rows : Integer. optional
        Number of image rows. The default is 224.
    img_cols : Integer. optional
        Number of image columns. The default is 224.
    num_classes : Integer. optional
        Number of dataset classes. The default is 10.
    batch_size : Integer. optional
        Batch size. The default is 32.
    data_augmentation : Bool. optional
        Using data augmentation or not. The default is False.
    data_dir : String. optional
        The directory of Keras ImageDataGenerator target. The default is None.
    preprocessing_function : Callable function, optional
        The fucntion for input image preprocessing. The default is None.
    verbose: Integer.
        | The verbosity of dataset setup information
        | 2: Show setup process, dataset name and data shape/number
        | 1: Only show setup process and dataset name 
        | 0: Dont print info

    Returns
    -------
    x_train : Ndarray
        The training data images array.
    x_test : Ndarray
        The validation data images array.
    y_train : Ndarray
        The training data label.
    y_test : Ndarray
        The validation data label.
    class_indices : List or Ndarray
        The name of each class respect to their index.
    datagen : ImageDataGenerator.flow_from_directory
        The ImageDataGenerator generated dataset class for batch data accessing.
    input_shape : Tuple
        The shape of dataset image for DNN input.

    """
    if (dataset == "cifar10"):
        
        if verbose>0:
            print('Setup CIFAR-10 dataset...')

        num_classes = 10
        
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if verbose>1:
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
        
        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        input_shape=x_train.shape[1:]
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        if not data_augmentation:
            if verbose>1:
                print('Not using data augmentation.')
            datagen=None
            
        else:
            if verbose>1:
                print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
        
            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)
        
        class_indices=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    elif (dataset == "mnist"):

        if verbose>0:
            print('Setup MNIST dataset...')
        
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        if verbose>1:
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        datagen=None
        class_indices=np.arange(0,10)

    elif (dataset == "ImageDataGenerator"):
        
        if data_dir is None:
            raise NameError('Please specify the ImageDataGenerator directory')
        
        if verbose>0:
            print('Setup ImageDataGenerator custom dataset at %s ...' % data_dir)
        
        # input image dimensions
        
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_cols, img_rows)
        else:
            input_shape = (img_cols, img_rows, 3)
        
        if data_augmentation:
            if verbose>1:
                print('Using real-time data augmentation.')
            if preprocessing_function is not None:
                evaluation_datagen = ImageDataGenerator(
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                preprocessing_function=preprocessing_function)
            else:
                evaluation_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        else:
            if verbose>1:
                print('Not using data augmentation.')
            if preprocessing_function is not None:
                evaluation_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
            else:
                evaluation_datagen = ImageDataGenerator(rescale=1. / 255)
            
            
        datagen = evaluation_datagen.flow_from_directory(
            data_dir,
            target_size=(img_rows, img_cols),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False)
        
        x_train=None
        x_test=None
        y_train=None
        y_test=None
        class_indices=list(datagen.class_indices.keys())
        
    else:
        print("wrong dataset given.\nChoose between \'mnist\' or \'cifar10\' or \'ImageDataGenerator\'\n")

    return x_train, x_test, y_train, y_test, class_indices, datagen, input_shape
