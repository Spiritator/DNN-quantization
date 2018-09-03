# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:35:46 2018

@author: areslab007
"""

import h5py
import numpy as np

##########################################################
#                    SETUP PARAMETER                     #
##########################################################
original_weight_name = '../cifar10_4C2F_weight.h5'
quantized_weight_name = None
# specify the filename for quantized weight, if not type "None"
weight_bit_width = 8
weight_factorial_bit = 4
rounding_type = 'nearest'
# choose between 'nearest' , 'zero' , 'down'


#%%

def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data



o_weight_f = h5py.File(original_weight_name,'r')
if quantized_weight_name is None:
    quantized_weight_name=original_weight_name[:-3]+('_quantized_%dB%dI%dF.h5' % (weight_bit_width, weight_bit_width-weight_factorial_bit-1, weight_factorial_bit))
    q_weight_f = h5py.File(quantized_weight_name,'w')
else:
    q_weight_f = h5py.File(quantized_weight_name,'w')
    
    
if 'keras_version' in o_weight_f.attrs:
        original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
else:
    original_keras_version = '1'
if 'backend' in o_weight_f.attrs:
    original_backend = o_weight_f.attrs['backend'].decode('utf8')
else:
    original_backend = None
    

layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
filtered_layer_names = []
for layer_name in layer_names:
    o_group = o_weight_f[layer_name]
    weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
    if weight_names:
        filtered_layer_names.append(layer_name)
        
        
q_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in layer_names])
q_weight_f.attrs.create('backend',original_backend.encode('utf8'))
q_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))

for layer_iter, layer_name in enumerate(layer_names):
    o_group = o_weight_f[layer_name]
    weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
    weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
    quantized_layer = q_weight_f.create_group(layer_names[layer_iter])
    quantized_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
    quantized_sublayer = quantized_layer.create_group(layer_names[layer_iter])
    
    for weight_iter, weight_name in enumerate(weight_names):
        m = np.power(2,weight_factorial_bit)
        quantized_weight_value = weight_values[weight_iter] * m
        
        if rounding_type == 'nearest':
            quantized_weight_value = np.round(quantized_weight_value)
        elif rounding_type == 'zero':
            quantized_weight_value = np.trunc(quantized_weight_value)
        elif rounding_type == 'down':
            quantized_weight_value = np.floor(quantized_weight_value)
        else:
            print('Wrong Rounding Type\nChoose between \'nearest\' , \'zero\' , \'down\'')
            
        quantized_weight_value = np.clip(quantized_weight_value/m, -np.power(2,weight_bit_width-weight_factorial_bit-1), np.power(2,weight_bit_width-weight_factorial_bit-1)-np.power(0.5,weight_factorial_bit))
            
        quantized_sublayer.create_dataset(weight_name[len(layer_name)+1:],weight_values[weight_iter].shape,weight_values[weight_iter].dtype,quantized_weight_value)
    
    
o_weight_f.close()
q_weight_f.close()

#return quantized_weight_name

