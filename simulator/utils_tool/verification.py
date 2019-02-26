# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:40:43 2019

@author: Yung-Yu Tsai

View all the intermediate value of a inference result
"""

import keras
from keras.models import Model
import numpy as np

def view_intermediate(model,input_x):
    """View all the intermediate output of a DNN model

    # Arguments
        model: Keras Model. The model wanted to test.
        input_x: nparray. The preprocessed numpy array as the test input for DNN.

    # Returns
        A list of numpy array of the intermediate output.
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
    intermediate_output=intermediate_model.predict(input_x)
    
    intermediate_output=[input_x]+intermediate_output
    
    if batch_inference:
        return intermediate_output
    else:
        return [output[0] for output in intermediate_output]
                             
        
#    if batch_inference:
#        output_list=[input_x]
#    else:
#        output_list=[input_x[0]]
#    
#    for n_layer in range(1,num_layers):
#        print('evaluating layer %d/%d'%(n_layer+1,num_layers))
#        
#        intermediate_model=Model(inputs=model.input,outputs=model.layers[n_layer].output)
#    
#        intermediate_output=intermediate_model.predict(input_x)
#        
#        if batch_inference:
#            output_list.append(intermediate_output)
#        else:
#            output_list.append(intermediate_output[0])
            
#    return output_list
    