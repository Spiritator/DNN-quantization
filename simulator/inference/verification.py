# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:40:43 2019

@author: Yung-Yu Tsai

View all the intermediate value of a inference result
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

def view_intermediate(model,input_x):
    """View all the intermediate output of a DNN model

    Arguments
    ---------
    model: Keras Model. 
        The model wanted to test.
    input_x: Ndarray. 
        The preprocessed numpy array as the test input for DNN.

    Returns
    -------
    List of Ndarray 
        The intermediate output.
    """
    layer_info=model.layers
    num_layers=len(layer_info)
    batch_inference=True
    
    if len(input_x.shape)<len(model.input.shape):
        input_x=np.expand_dims(input_x, axis=0)
        batch_inference=False
        
    output_list=list()
    
    for n_layer in range(1,num_layers):
        output_list.append(model.layers[n_layer].output)
        
    print('building verification model...')
    intermediate_model=Model(inputs=model.input,outputs=output_list)
    
    print('predicting...')
    intermediate_output=intermediate_model.predict(input_x,verbose=True)
    
    intermediate_output=[input_x]+intermediate_output
    
    if batch_inference:
        return intermediate_output
    else:
        return [output[0] for output in intermediate_output]
                 
def view_fmap_distribution_batch(model,input_x=None, observe_layer_idxs=None, 
                                 num_quantiles=None, bins=None):
    """ View feature map distribution for a single batch of sample
        The statistical data will be produce by the entire batch
    
    Parameters
    ----------
    model : tensorlow.keras.model
        The model that are being viewed for distribution.
    input_x : Ndarray, optional
        The input dataset for evaluation as the reference for feature map distributions.
        Assume the input_x array are preprocessed images.
    observe_layer_idxs: List of Integer.
        The indexes of layers that are the subjects which user wanted to view their feature map distribution.
        If None, all layers will get its distribution report which is not the common case.
        If List are given, only the targeted layers will have distribution information dictionary, others set to None.
        
    num_quantiles: Integer. 
        The number of intervals the returned num_quantiles + 1 cut points divide the range into.
    bins: Integer
        The number of bins for layer weight histogram inspection.

    Returns
    -------
    model_fmap_distribution: List of Dictionary
        The feature map distribution information for given model. List index is the same as layer index in model.
        Each element contains statistic information.
        
        >>> [{'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L1 ifmap
        ...  {'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L2 ifmap
        ...  --- ]


    """
    if num_quantiles is None:
        num_quantiles=10
    if bins is None:
        bins=100
            
    model_depth=len(model.layers)
    layer_names=[l.name for l in model.layers]
    model_fmap_distribution = [None for i in range(model_depth)]
    if observe_layer_idxs is None:
        observe_layer_idxs=range(model_depth)
    
    # build statistic model
    output_list=list()
    for n_layer in observe_layer_idxs:
        fmap=model.layers[n_layer].input
        mean=tf.reduce_mean(fmap)
        stddev=tf.math.reduce_std(fmap)
        bin_edges=tf.linspace(tf.reduce_min(fmap),tf.reduce_max(fmap),bins+1)
        hist=tfp.stats.histogram(fmap,bin_edges)
        quantile_values=tfp.stats.quantiles(fmap,num_quantiles)
        
        dist_info=[tf.stack([mean,stddev]),hist,bin_edges,quantile_values]
        output_list.append(dist_info)
        
    print('building statistic model...')
    statistic_model=Model(inputs=model.input,outputs=output_list)   
    
    distribution_list=statistic_model.predict(input_x)
    
    return distribution_list
             
        
def view_fmap_distribution(model,input_x=None, batch_size=None, datagen=None, observe_layer_idxs=None, 
                           num_quantiles=None, bins=None):
    """ View feature map distribution for a dataset
        The statistical data will be produce in batch wise feature maps per step.
        Then get the cummulative moving average of each statistic through all steps.
    
    Parameters
    ----------
    model : tensorlow.keras.model
        The model that are being viewed for distribution.
    input_x : Ndarray, optional
        The input dataset for evaluation as the reference for feature map distributions.
        Assume the input_x array are preprocessed images.
    batch_size : Integer, optional
        The batch size of dataset split. 
    datagen : tensorflow.keras.preprocessing.image.ImageDataGenerator.flow, optional. Overwrite input_x.
        The flowed Keras ImageDataGenerator. This means the evaluate dataset has been preprocessed and batch grouped.
        The statistic analysis will base on the batch size set in ImageDataGenerator flow. The default is None.
    observe_layer_idxs: List of Integer.
        The indexes of layers that are the subjects which user wanted to view their feature map distribution.
        If None, all layers will get its distribution report which is not the common case.
        If List are given, only the targeted layers will have distribution information dictionary, others set to None.
        
    num_quantiles: Integer. 
        The number of intervals the returned num_quantiles + 1 cut points divide the range into.
    bins: Integer
        The number of bins for layer weight histogram inspection.

    Returns
    -------
    model_fmap_distribution: List of Dictionary
        The feature map distribution information for given model. List index is the same as layer index in model.
        Each element contains statistic information.
        
        >>> [{'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L1 ifmap
        ...  {'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L2 ifmap
        ...  --- ]


    """
    if num_quantiles is None:
        num_quantiles=10
    if bins is None:
        bins=100
    
    if input_x is None and datagen is None:
        raise ValueError('Both input_x and datagen are None, atleast have one input type for model.')
        
    if datagen is None:
        datagen=ImageDataGenerator()
        datagen=datagen.flow(input_x,batch_size=batch_size)
        
    is_label=isinstance(datagen[0],tuple)
        
    model_depth=len(model.layers)
    layer_names=[l.name for l in model.layers]
    model_fmap_distribution = [None for i in range(model_depth)]
    if observe_layer_idxs is None:
        observe_layer_idxs=range(model_depth)
    
    # build statistic model
    output_list=list()
    for n_layer in observe_layer_idxs:
        fmap=model.layers[n_layer].input
        mean=tf.reduce_mean(fmap)
        stddev=tf.math.reduce_std(fmap)
        bin_edges=tf.linspace(tf.reduce_min(fmap),tf.reduce_max(fmap),bins+1)
        hist=tfp.stats.histogram(fmap,bin_edges)
        quantile_values=tfp.stats.quantiles(fmap,num_quantiles)
        
        dist_info=[mean,stddev,hist,bin_edges,quantile_values]
        output_list.append(dist_info)
        
    print('building statistic model...')
    statistic_model=Model(inputs=model.input,outputs=output_list)

    


    