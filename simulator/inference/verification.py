# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:40:43 2019

@author: Yung-Yu Tsai

View all the intermediate value of a inference result
"""

from tensorflow.keras.models import Model
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
                             
        
def view_fmap_distribution(model,input_x,batch_size=None, datagen=None):
    """ View feature map distribution 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    input_x : TYPE
        DESCRIPTION.
    batch_size : TYPE, optional
        DESCRIPTION. The default is None.
    datagen : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    