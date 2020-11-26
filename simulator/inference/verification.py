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
import tqdm as tqdm

def view_intermediate(model,input_x,eager_mode=False):
    """View all the intermediate output of a DNN model

    Arguments
    ---------
    model: Keras Model. 
        The model wanted to test.
    input_x: Ndarray. 
        The preprocessed numpy array as the test input for DNN.
    eager_mode: Bool. Default is False
        Execute verification eagerly or not. The eager mode and graph mode of TensorFlow may got different result.
        Eager mode got the same result as RTL verilog implementation.

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
    if eager_mode:
        intermediate_output=intermediate_model(input_x)
        intermediate_output=[output.numpy() for output in intermediate_output]
    else:
        intermediate_output=intermediate_model.predict(input_x,verbose=True)
    
    intermediate_output=[input_x]+intermediate_output
    
    if batch_inference:
        return intermediate_output
    else:
        return [output[0] for output in intermediate_output]
                 
def _build_intermediate_model(model,observe_layer_idxs):
    """ Build model with observed layer input feature maps as output list """
    output_list=list()
    
    for n_layer in observe_layer_idxs:
        fmap=model.layers[n_layer].input
        output_list.append(fmap)
        
    intermediate_model=Model(inputs=model.input,outputs=output_list)   
    
    return intermediate_model

@tf.function
def fmap_statistic(fmap, num_quantiles=None, bins=None):
    """ Make statistic analysis on given fmap data with tf.funtion

    Parameters
    ----------
    fmap : Ndarray
        The input feature maps of targeted observe layer. 
    num_quantiles: Integer. 
        The number of intervals the returned num_quantiles + 1 cut points divide the range into.
    bins: Integer
        The number of bins for layer weight histogram inspection.

    Returns
    -------
    mean : Float
        The mean of fmap.
    stddev : Float
        The standard deviation of fmap.
    hist : Ndarray (Integer)
        The count of appearence in histogram value section.
    bin_edges : Ndarray (Float)
        The value of histogram section edges.
    quantile_values : Ndarray (Float)
        The data value of quantile points.

    """
    mean=tf.reduce_mean(fmap)
    stddev=tf.math.reduce_std(fmap)
    bin_edges=tf.linspace(tf.reduce_min(fmap),tf.reduce_max(fmap),bins+1)
    hist=tfp.stats.histogram(fmap,bin_edges)
    quantile_values=tfp.stats.quantiles(fmap,num_quantiles)

    return mean, stddev, hist, bin_edges, quantile_values
    
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
        
        >>> [{'mean':average,                  #L1 ifmap
        ...   'std_dev':standard_deviation,
        ...   'hist':histogram_count,
        ...   'bin_edges':histogram_bin_egdes,
        ...   'quantile':quantile_data_value},
        ...  {'mean':average,                  #L2 ifmap
        ...   'std_dev':standard_deviation,
        ...   'hist':histogram_count,
        ...   'bin_edges':histogram_bin_egdes,
        ...   'quantile':quantile_data_value},
        ...  --- ]                             # Layer N ifmap


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
    
    print('building statistic model...')
    statistic_model=_build_intermediate_model(model,observe_layer_idxs)
    
    ifmap_list=statistic_model.predict(input_x)
    
    for idx,fmap in enumerate(ifmap_list):
        stats_tmp=fmap_statistic(fmap,num_quantiles,bins)
        dist_tmp={'layer_name':layer_names[observe_layer_idxs[idx]],
                  'mean':stats_tmp[0].numpy(),
                  'std_dev':stats_tmp[1].numpy(),
                  'hist':stats_tmp[2].numpy(),
                  'bin_edges':stats_tmp[3].numpy(),
                  'quantile':stats_tmp[4].numpy()}
        model_fmap_distribution[observe_layer_idxs[idx]]=dist_tmp

    return model_fmap_distribution
             
        
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
        
        >>> [{'mean':average,                  #L1 ifmap
        ...   'std_dev':standard_deviation,
        ...   'hist':histogram_count,
        ...   'bin_edges':histogram_bin_egdes,
        ...   'quantile':quantile_data_value},
        ...  {'mean':average,                  #L2 ifmap
        ...   'std_dev':standard_deviation,
        ...   'hist':histogram_count,
        ...   'bin_edges':histogram_bin_egdes,
        ...   'quantile':quantile_data_value},
        ...  --- ]                             # Layer N ifmap


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
    num_steps=len(datagen)
        
    is_label=isinstance(datagen[0],tuple)
        
    model_depth=len(model.layers)
    layer_names=[l.name for l in model.layers]
    model_fmap_distribution = [None for i in range(model_depth)]
    if observe_layer_idxs is None:
        observe_layer_idxs=range(model_depth)
    for i in observe_layer_idxs:
        model_fmap_distribution[i]={'layer_name':layer_names[i],
                                    'mean':0.0,
                                    'std_dev':0.0,
                                    'hist':np.zeros(bins,dtype=np.float32),
                                    'bin_edges':np.zeros(bins+1,dtype=np.float32),
                                    'quantile':np.zeros(num_quantiles+1,dtype=np.float32)}
    
    print('building statistic model...')
    statistic_model=_build_intermediate_model(model,observe_layer_idxs)
    
    pbar=tqdm.tqdm(desc='Steps', total=num_steps)
    cnt=0
    for data in datagen:
        if is_label:
            data=data[0]
            
        ifmap_list=statistic_model.predict(data)
        
        for idx,fmap in enumerate(ifmap_list):
            stats_tmp=fmap_statistic(fmap,num_quantiles,bins)
            model_fmap_distribution[observe_layer_idxs[idx]]['mean']+=stats_tmp[0].numpy()
            model_fmap_distribution[observe_layer_idxs[idx]]['std_dev']+=stats_tmp[1].numpy()
            model_fmap_distribution[observe_layer_idxs[idx]]['hist']+=stats_tmp[2].numpy()
            model_fmap_distribution[observe_layer_idxs[idx]]['bin_edges']+=stats_tmp[3].numpy()
            model_fmap_distribution[observe_layer_idxs[idx]]['quantile']+=stats_tmp[4].numpy()
        
        pbar.update()
        cnt+=1
        if cnt==num_steps:
            break
    pbar.close()
        
    for i in observe_layer_idxs:
        model_fmap_distribution[i]['mean']/=num_steps
        model_fmap_distribution[i]['std_dev']/=num_steps
        model_fmap_distribution[i]['hist']/=num_steps
        model_fmap_distribution[i]['bin_edges']/=num_steps
        model_fmap_distribution[i]['quantile']/=num_steps

    return model_fmap_distribution

    


    