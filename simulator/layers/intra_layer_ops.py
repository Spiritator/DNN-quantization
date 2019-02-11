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
def q2dconvolution_op(inputs, filters, quantizer, strides, rate, padding, data_format):
    ''' Reimplementation of the 2D convolution layer.
    Args: 
        inputs:  [batch_size, image_height, image_width, input_channels] 
        filters: [filter_height, filter_width, input_channels, output_channels]
        quantizer: Quantizer object, has interface '.quantize(tensor)'       
    '''
    PARALLEL_ITERATIONS=1 # number of convolution ops which can run in parallel.

    if data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(data_format))
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value
    output = tf.split(inputs,batch_size)

    # prepare filters
    filter_shape = filters.get_shape()
    filters=tf.split(filters,filters.shape.dims[3].value,axis=3)

    # get patch shape, needed for result shape estimation
    patch = tf.extract_image_patches(output[0], 
                                           ksizes=(1,filter_shape.dims[0], filter_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
    patch_shape = patch.get_shape()

    # inner loop condition and body.
    # iterates over all output maps
    def inner_cond(index, outputs, output_patch):
        return index < filter_shape.dims[3].value 

    def inner_body(index, outputs, output_patch):
        current_filter = tf.gather(filters, index)
        current_filter = tf.reshape(current_filter, [1,1,1,patch_shape.dims[3].value])
        current_filter = tf.tile(current_filter,[1,patch_shape.dims[1].value,patch_shape.dims[2].value,1])  
        out = tf.multiply(output_patch, current_filter)
        if quantizer is not None:
            out = quantizer.quantize(out)     # quantize after multiply
        out = tf.reduce_sum(out,axis=3,keep_dims=True)
        if quantizer is not None:
            out = quantizer.quantize(out)     # quantize after add
        outputs = tf.concat([outputs,out],3)
        return [tf.add(index,1), outputs, output_patch]

    # outer loop condition and body
    # iterates over all batches
    def outer_cond(batch, result):
        return batch < batch_size

    def outer_body(batch, result):
        # extract patch form global 'output'
        output_patch = tf.extract_image_patches(tf.gather(output,batch), 
                                           ksizes=(1,filter_shape.dims[0], filter_shape.dims[1],1), 
                                           strides=strides,
                                           rates=rate,#[1,1,1,1],
                                           padding=padding )
        # prepare inner loop interation variable 'out_filter'
        out_filter=tf.constant(0)
        # placeholder 'outputs', results will be concatenated to this tensor. 
        # Remove first element after all elements are computed!
        outputs=tf.constant(0.0,
                            shape=[1, output_patch.shape.dims[1].value,
                            output_patch.shape.dims[2].value, 1])
        # start inner loop. pass loop iterator, result placeholder and patch. 
        # Take 2nd element [1] as result!
        outputs=tf.while_loop( inner_cond, inner_body, [out_filter, outputs, output_patch],
                shape_invariants=[ out_filter.get_shape(), tf.TensorShape(
                    [1,output_patch.shape.dims[1].value,output_patch.shape.dims[2].value,None]),
                    output_patch.get_shape() ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
        # concatenate batches (along axis 0).
        # remove first placeholder element from outputs!
        result= tf.concat([ result,outputs[:,:,:,1:] ], 0)
        return [tf.add(batch,1), result]
    
    # main
    # prepare outer loop iteration variable 'batch'
    batch=tf.constant(0)
    # placeholder 'result', results from inner loop will be concatenated to this tensor.
    result= tf.constant( 0.0,
                          shape=[1, patch_shape.dims[1].value,
                          patch_shape.dims[2].value, filter_shape.dims[3].value] )
    # start outer loop. pass 'batch' and 'result'.
    # Take 2nd element [1] as result!
    result = tf.while_loop( outer_cond, outer_body, [batch, result],
                shape_invariants=[ batch.get_shape(), tf.TensorShape(
                    [None,patch_shape.dims[1].value,patch_shape.dims[2].value,filter_shape.dims[3]]) ],
                parallel_iterations=PARALLEL_ITERATIONS,
                swap_memory=True )[1]
    # remove first element from placeholder!
    output = result[1:,:,:,:]
    # output = tf.squeeze(tf.stack(output),axis=[1])

    # setting shape, since partially ignored by while_loops
    output.set_shape([batch_size, 
                        output.shape.dims[1].value,
                        output.shape.dims[2].value,
                        filter_shape.dims[3].value]) 
    return output

# quantized batch normalization calculation
# tensorflow/python/ops/nn_impl.py
def qbatch_normalization(inputs,
                        mean,
                        variance,
                        beta,
                        gamma,
                        variance_epsilon,
                        quantizer, 
                        nb, 
                        fb, 
                        rounding_method,
                        name=None):
  r"""Batch normalization.
  As described in http://arxiv.org/abs/1502.03167.
  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
  \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:
    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
      thereof during inference.
  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).
  Returns:
    the normalized, scaled, offset tensor.
  """
  with ops.name_scope(name, "batchnorm", [inputs, mean, variance, gamma, beta]):
    #internal calculation of sqrt is NOT quantized! 
    coef = quantize( math_ops.sqrt(variance + variance_epsilon), nb, fb, rounding_method )
    coef = quantize( math_ops.reciprocal(coef), nb, fb, rounding_method )
    if gamma is not None:
      coef = quantize(coef*gamma, nb, fb, rounding_method)
    
    if beta is not None:
        const = quantize( beta - quantizer.quantize(mean * coef), nb, fb, rounding_method )
    else:
        const = quantize(-mean * coef, nb, fb, rounding_method)
    result = quantize( quantize(inputs * coef, nb, fb, rounding_method) + const, nb, fb, rounding_method )
    return result

