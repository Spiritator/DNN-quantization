# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:32:01 2019
refernce: https://github.com/cc-hpc-itwm/TensorQuant
all the credit refer to TensorQuant, available https://arxiv.org/abs/1710.05758

@author: Yung-Yu Tsai
"""

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from layers.quantized_ops import quantize

def QuantizedDenseCore(inputs, kernel, nb, fb, rounding_method):
    ''' Reimplementation of the Dense layer.
    Args: 
        inputs:  [batch_size, neurons] 
        kernel: [input_neurons, output_neurons]
    '''    
    
    PARALLEL_ITERATIONS=4 # number of convolution ops which can run in parallel.
    
    batch_size = inputs.shape.dims[0].value  
    input_size = inputs.shape.dims[1].value
    output_size = kernel.get_shape().dims[1].value

    # work around of tf.slice bug in multi gpu condition
    if batch_size is None:
        batch_size=tf.shape(inputs)[:1]
        output=tf.reshape(inputs,shape=[-1,1,input_size])
    else:
        output = tf.split(inputs,batch_size)
    
    # work around of tf.slice bug in multi gpu condition
    if not isinstance(batch_size,int):
        batch_size=batch_size[0]
        
        
    def batch_cond(batch, neurons):
        return batch < batch_size

    def batch_body(batch, neurons):
        output_tmp = tf.gather(output,batch)
        output_tmp = tf.reshape(output_tmp,[input_size,1])
        output_tmp = tf.tile(output_tmp,[1,output_size])
        
        output_tmp = tf.multiply(output_tmp,kernel)
        # quantize after multiplication
        output_tmp = quantize(output_tmp, nb=nb, fb=fb, rounding_method=rounding_method) 
        
        output_tmp = tf.reduce_sum(output_tmp,axis=0,keepdims=True)
        # quantize after accumulation
        output_tmp = quantize(output_tmp, nb=nb, fb=fb, rounding_method=rounding_method) 
        # concatenate batches (along axis 0).
        neurons= tf.concat([ neurons,output_tmp], 0)
        return [tf.add(batch,1), neurons]
        
    # prepare outer loop iteration variable 'batch'
    batch = tf.constant(0)
    # placeholder 'ofmap', ofmaps from inner loop will be concatenated to this tensor.
    neurons = tf.constant( 0.0, shape=[1, output_size] )
    # start loop. pass 'batch' and 'ofmap'.
    # Take 2nd element [1] as ofmap!
    neurons = tf.while_loop( batch_cond, batch_body, [batch, neurons],
                shape_invariants=[ batch.get_shape(), tf.TensorShape(
                    [None,output_size]) ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
    # remove first element from placeholder!
    output = neurons[1:]
    
    output = tf.reshape(output,[batch_size,output_size])
    
    return output


##########################
### Reimplemented Conv ###
##########################
# parallel_iterations and swap_memory in tf.while_loops can be adjusted
def QuantizedConv2DCore(inputs, kernel, strides, rate, padding, data_format, nb, fb, rounding_method):
    ''' Reimplementation of the 2D convolution layer.
    Args: 
        inputs:  [batch_size, image_height, image_width, input_channels] 
        kernel: [kernel_height, kernel_width, input_channels, output_channels]
    '''
    PARALLEL_ITERATIONS=4 # number of convolution ops which can run in parallel.

    if data_format not in ("channels_last", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(data_format))
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value

    # work around of tf.slice bug in multi gpu condition
    if batch_size is None:
        batch_size=tf.shape(inputs)[:1]
        output=tf.reshape(inputs,[-1,1,inputs.shape.dims[1].value,inputs.shape.dims[2].value,inputs.shape.dims[3].value])
    else:
        output = tf.split(inputs,batch_size)
    
    # work around of tf.slice bug in multi gpu condition
    if not isinstance(batch_size,int):
        batch_size=batch_size[0]

    # prepare kernel
    kernel_shape = kernel.get_shape()
    kernel = tf.split(kernel,kernel.shape.dims[3].value,axis=3)

    # get patch shape, needed for ofmap shape estimation
    patch = tf.extract_image_patches(output[0], 
                                           ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
    patch_shape = patch.get_shape()
    #[input channel, ofmap height, ofmap width, num of kernel psum * input channel]

    # inner loop condition and body.
    # iterates over all output maps
    def inner_cond(index, outputs, output_patch):
        return index < kernel_shape.dims[3].value 

    def inner_body(index, outputs, output_patch):
        kernel_tmp = tf.gather(kernel, index)
        kernel_tmp = tf.reshape(kernel_tmp, [1,1,1,patch_shape.dims[3].value])
        kernel_tmp = tf.tile(kernel_tmp,[1,patch_shape.dims[1].value,patch_shape.dims[2].value,1])  
        
        out_tmp = tf.multiply(output_patch, kernel_tmp)
        # quantize after multiplication
        out_tmp = quantize(out_tmp, nb, fb, rounding_method)     
        
        out_tmp = tf.reduce_sum(out_tmp,axis=3,keepdims=True)
        # quantize after accumulation
        out_tmp = quantize(out_tmp, nb, fb, rounding_method)     
        
        outputs = tf.concat([outputs,out_tmp],3)
        
        return [tf.add(index,1), outputs, output_patch]

    # outer loop condition and body
    # iterates over all batches
    def outer_cond(batch, ofmap):
        return batch < batch_size

    def outer_body(batch, ofmap):
        # extract patch form global 'output'
        output_patch = tf.extract_image_patches(tf.gather(output,batch), 
                                           ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
        # prepare inner loop interation variable 'out_kernel'
        out_kernel=tf.constant(0)
        # placeholder 'outputs', ofmaps will be concatenated to this tensor. 
        # Remove first element after all elements are computed!
        outputs=tf.constant(0.0,
                            shape=[1, output_patch.shape.dims[1].value,
                            output_patch.shape.dims[2].value, 1])
        # start inner loop. pass loop iterator, ofmap placeholder and patch. 
        # Take 2nd element [1] as ofmap!
        outputs=tf.while_loop( inner_cond, inner_body, [out_kernel, outputs, output_patch],
                shape_invariants=[ out_kernel.get_shape(), tf.TensorShape(
                    [1,output_patch.shape.dims[1].value,output_patch.shape.dims[2].value,None]),
                    output_patch.get_shape() ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
        # concatenate batches (along axis 0).
        # remove first placeholder element from outputs!
        ofmap= tf.concat([ ofmap,outputs[:,:,:,1:] ], 0)
        return [tf.add(batch,1), ofmap]
    
    # main
    # prepare outer loop iteration variable 'batch'
    batch=tf.constant(0)
    # placeholder 'ofmap', ofmaps from inner loop will be concatenated to this tensor.
    ofmap= tf.constant( 0.0,
                          shape=[1, patch_shape.dims[1].value,
                          patch_shape.dims[2].value, kernel_shape.dims[3].value] )
    # start outer loop. pass 'batch' and 'ofmap'.
    # Take 2nd element [1] as ofmap!
    ofmap = tf.while_loop( outer_cond, outer_body, [batch, ofmap],
                shape_invariants=[ batch.get_shape(), tf.TensorShape(
                    [None,patch_shape.dims[1].value,patch_shape.dims[2].value,kernel_shape.dims[3]]) ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
    # remove first element from placeholder!
    output = ofmap[1:,:,:,:]

    # setting shape, since partially ignored by while_loops
    output = tf.reshape(output,[batch_size, 
                        output.shape.dims[1].value,
                        output.shape.dims[2].value,
                        kernel_shape.dims[3].value]) 

#    output.set_shape([batch_size[0], 
#                        output.shape.dims[1].value,
#                        output.shape.dims[2].value,
#                        kernel_shape.dims[3].value]) 
    return output

# quantized batch normalization calculation
# tensorflow/python/ops/nn_impl.py
def QuantizedBatchNormalizationCore(inputs,
                                    mean,
                                    variance,
                                    beta,
                                    gamma,
                                    variance_epsilon,
                                    nb, 
                                    fb, 
                                    rounding_method,
                                    name=None):
    with ops.name_scope(name, "batchnorm", [inputs, mean, variance, gamma, beta]):
        coef = quantize( math_ops.sqrt(variance + variance_epsilon), nb, fb, rounding_method )
        coef = quantize( math_ops.reciprocal(coef), nb, fb, rounding_method )
        if gamma is not None:
          coef = quantize(coef*gamma, nb, fb, rounding_method)
        
        if beta is not None:
            const = quantize( beta - quantize(mean * coef, nb, fb, rounding_method), nb, fb, rounding_method )
        else:
            const = quantize(-mean * coef, nb, fb, rounding_method)
        output = quantize( quantize(inputs * coef, nb, fb, rounding_method) + const, nb, fb, rounding_method )
        return output


###########################################
### Reimplemented Depthwise Convolution ###
###########################################
# parallel_iterations and swap_memory in tf.while_loops can be adjusted
def QuantizedDepthwiseConv2DCore(inputs, kernel, strides, rate, padding, data_format, nb, fb, rounding_method):
    ''' Reimplementation of the 2D depthwise convolution layer.
    Args: 
        inputs:  [batch_size, image_height, image_width, input_channels] 
        kernel: [kernel_height, kernel_width, input_channels, output_channels]
    '''
    PARALLEL_ITERATIONS=4 # number of convolution ops which can run in parallel.

    if data_format not in ("channels_last", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(data_format))
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value
    
    # work around of tf.slice bug in multi gpu condition
    if batch_size is None:
        batch_size=tf.shape(inputs)[:1]
        output=tf.reshape(inputs,[-1,1,inputs.shape.dims[1].value,inputs.shape.dims[2].value,inputs.shape.dims[3].value])
    else:
        output = tf.split(inputs,batch_size)
    
    # work around of tf.slice bug in multi gpu condition
    if not isinstance(batch_size,int):
        batch_size=batch_size[0]


    # prepare kernel
    kernel_shape = kernel.get_shape()
    #kernel = tf.split(kernel,kernel.shape.dims[3].value,axis=3)
    # dont need in depthwise conv2D

    # get patch shape, needed for ofmap shape estimation
    patch = tf.extract_image_patches(output[0], 
                                           ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
    patch_shape = patch.get_shape()
    #[input channel, ofmap height, ofmap width, num of kernel psum * input channel]

    # inner body depthwise convolution

    def inner_body(output_patch):
        kernel_tmp = tf.reshape(kernel, [1,1,1,patch_shape.dims[3].value])
        kernel_tmp = tf.tile(kernel_tmp,[1,patch_shape.dims[1].value,patch_shape.dims[2].value,1])  
        
        out_tmp = tf.multiply(output_patch, kernel_tmp)
        # quantize after multiplication
        out_tmp = quantize(out_tmp, nb, fb, rounding_method)    
        
        out_tmp = tf.reshape(out_tmp, [1,patch_shape.dims[1].value,patch_shape.dims[2].value,tf.reduce_prod(kernel_shape[0:2]),kernel_shape.dims[2].value])
        
        out_tmp = tf.reduce_sum(out_tmp,axis=3,keepdims=False)
        # quantize after accumulation
        out_tmp = quantize(out_tmp, nb, fb, rounding_method)     
                
        return out_tmp

    # outer loop condition and body
    # iterates over all batches
    def outer_cond(batch, ofmap):
        return batch < batch_size

    def outer_body(batch, ofmap):
        # extract patch form global 'output'
        output_patch = tf.extract_image_patches(tf.gather(output,batch), 
                                           ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
        # start inner loop. pass loop iterator, ofmap placeholder and patch. 
        # Take 2nd element [1] as ofmap!
        outputs=inner_body(output_patch)
        # concatenate batches (along axis 0).
        # remove first placeholder element from outputs!
        ofmap= tf.concat([ ofmap,outputs ], 0)
        return [tf.add(batch,1), ofmap]
    
    # main
    # prepare outer loop iteration variable 'batch'
    batch=tf.constant(0)
    # placeholder 'ofmap', ofmaps from inner loop will be concatenated to this tensor.
    ofmap= tf.constant( 0.0,
                          shape=[1, patch_shape.dims[1].value,
                          patch_shape.dims[2].value, kernel_shape.dims[2].value] )
    # start outer loop. pass 'batch' and 'ofmap'.
    # Take 2nd element [1] as ofmap!
    ofmap = tf.while_loop( outer_cond, outer_body, [batch, ofmap],
                shape_invariants=[ batch.get_shape(), tf.TensorShape(
                    [None,patch_shape.dims[1].value,patch_shape.dims[2].value,kernel_shape.dims[2]]) ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
    # remove first element from placeholder!
    output = ofmap[1:,:,:,:]

    # setting shape, since partially ignored by while_loops
    output = tf.reshape(output,[batch_size, 
                        output.shape.dims[1].value,
                        output.shape.dims[2].value,
                        kernel_shape.dims[2].value]) 

#    output.set_shape([batch_size, 
#                        output.shape.dims[1].value,
#                        output.shape.dims[2].value,
#                        kernel_shape.dims[2].value]) 
    return output
