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

from layers.quantized_ops import quantize, clip_through

def QuantizedDenseCore(inputs, kernel, nb, fb, rounding_method):
    if isinstance(inputs,list):
        # for the bypass method of keras flatten layer batch number bug
        batch_size=inputs[1]
        inputs=inputs[0]
#    else:
#        raise TypeError('wrong type of input being assigned. The input is either Tensor (normal injection) or list (index 0 fault list, index 1 the data shape of being injected tensor.)')
    else:
        batch_size = inputs.shape.dims[0].value
        
    input_size = inputs.shape.dims[1].value
    output_size = kernel.get_shape().dims[1].value
    output = tf.split(inputs,batch_size)
    for batch in range(batch_size):
        output_tmp = output[batch]
        output_tmp = tf.reshape(output_tmp,[input_size,1])
        output_tmp = tf.tile(output_tmp,[1,output_size])
        
        output_tmp = tf.multiply(output_tmp,kernel)
        # quantize after multiplication
        output_tmp = quantize(output_tmp, nb=nb, fb=fb, rounding_method=rounding_method) 
        
        output_tmp = tf.reduce_sum(output_tmp,axis=[0])
        # quantize after accumulation
        output_tmp = quantize(output_tmp, nb=nb, fb=fb, rounding_method=rounding_method) 
        output[batch] = output_tmp
        
    output = tf.stack(output)
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
        quantizer: Quantizer object, has interface '.quantize(tensor)'       
    '''
    PARALLEL_ITERATIONS=1 # number of convolution ops which can run in parallel.

    if data_format not in ("channels_last", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(data_format))
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value
    output = tf.split(inputs,batch_size)

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
    output.set_shape([batch_size, 
                        output.shape.dims[1].value,
                        output.shape.dims[2].value,
                        kernel_shape.dims[3].value]) 
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

