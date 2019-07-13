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

import keras.backend as K

PARALLEL_ITERATIONS=4 # number of convolution ops which can run in parallel.
tf_while_loop=False

def _preprocess_conv2d_input(x, data_format):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """
    if K.dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    tf_data_format = 'NHWC'
    if data_format == 'channels_first':
        tf_data_format = 'NCHW'
    return x, tf_data_format

def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.

    # Arguments
        padding: string, `"same"` or `"valid"`.

    # Returns
        a string, `"SAME"` or `"VALID"`.

    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    elif padding == 'SAME' or padding == 'VALID':
        pass
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding



def QuantizedDenseCore(inputs, kernel, Q_info):
    ''' Reimplementation of the Dense layer.
    Args: 
        inputs:  [batch_size, neurons] 
        kernel: [input_neurons, output_neurons]
    '''    
        
    batch_size = inputs.shape.dims[0].value  
    input_size = inputs.shape.dims[1].value
    output_size = kernel.get_shape().dims[1].value

    if not tf_while_loop:
        # work around of tf.slice bug in multi gpu condition
        if batch_size is None:
            batch_size = tf.shape(inputs)[:1]
    
        output = tf.expand_dims(inputs,axis=2)
        output = tf.tile(output,[1,1,output_size])
        
        # work around of tf.slice bug in multi gpu condition
        if not isinstance(batch_size,int):
            batch_size = batch_size[0]
            
        kernel_tmp = tf.expand_dims(kernel,axis=0)
        kernel_tmp = tf.tile(kernel_tmp,[batch_size,1,1])
        
        output = tf.multiply(output,kernel_tmp)
            
        # quantize after multiplication
        output = Q_info.quantize(output) 
        
        output = tf.reduce_sum(output,axis=1,keepdims=False)
        # quantize after accumulation
        output = Q_info.quantize(output) 
    
    else:
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
            output_tmp = Q_info.quantize(output_tmp) 
            
            output_tmp = tf.reduce_sum(output_tmp,axis=0,keepdims=True)
            # quantize after accumulation
            output_tmp = Q_info.quantize(output_tmp) 
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
def QuantizedConv2DCore(inputs, kernel, strides, rate, padding, data_format, Q_info):
    ''' Reimplementation of the 2D convolution layer.
    Args: 
        inputs:  [batch_size, image_height, image_width, input_channels] 
        kernel: [kernel_height, kernel_width, input_channels, output_channels]
    '''

    inputs, tf_data_format = _preprocess_conv2d_input(inputs, data_format)
    if tf_data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(tf_data_format))
    
    padding = _preprocess_padding(padding)
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value

    if not tf_while_loop:
        # work around of tf.slice bug in multi gpu condition
        if batch_size is None:
            batch_size=tf.shape(inputs)[:1]
        
        # work around of tf.slice bug in multi gpu condition
        if not isinstance(batch_size,int):
            batch_size=batch_size[0]
    
        # prepare kernel
        kernel_shape = kernel.get_shape()
        #kernel = tf.split(kernel,kernel.shape.dims[3].value,axis=3)
    
        # get output for conv multiply
        output = tf.extract_image_patches(inputs, 
                                          ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                          strides=strides,
                                          rates=rate,#[1,1,1,1],
                                          padding=padding )
        patch_shape = output.get_shape()
        #[batch, ofmap height, ofmap width, num of kernel psum * input channel]
        
        output = tf.expand_dims(output,axis=-1)
        output = tf.tile(output,[1,1,1,1,kernel_shape.dims[3].value])
        #[batch, ofmap height, ofmap width, num of kernel psum * input channel, output channel]
        
        kernel_tmp = tf.reshape(kernel, [1,1,1,patch_shape.dims[3].value,kernel_shape.dims[3].value])
        kernel_tmp = tf.tile(kernel_tmp,[batch_size,patch_shape.dims[1].value,patch_shape.dims[2].value,1,1])  
        #[batch, ofmap height, ofmap width, num of kernel psum * input channel, output channel]
    
        output = tf.multiply(output, kernel_tmp)
        # quantize after multiplication
        output = Q_info.quantize(output)     
        
        output = tf.reduce_sum(output,axis=3,keepdims=False)
        # quantize after accumulation
        output = Q_info.quantize(output)     
    
    else:
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
            out_tmp = Q_info.quantize(out_tmp)     
            
            out_tmp = tf.reduce_sum(out_tmp,axis=3,keepdims=True)
            # quantize after accumulation
            out_tmp = Q_info.quantize(out_tmp)     
            
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


    return output

# quantized batch normalization calculation
# tensorflow/python/ops/nn_impl.py
def QuantizedBatchNormalizationCore(inputs,
                                    mean,
                                    variance,
                                    beta,
                                    gamma,
                                    variance_epsilon,
                                    Q_info,
                                    name=None):
    with ops.name_scope(name, "batchnorm", [inputs, mean, variance, gamma, beta]):
        coef = Q_info.quantize( math_ops.sqrt(variance + variance_epsilon))
        coef = Q_info.quantize( math_ops.reciprocal(coef))
        if gamma is not None:
          coef = Q_info.quantize(coef*gamma)
        
        if beta is not None:
            const = Q_info.quantize( beta - Q_info.quantize(mean * coef))
        else:
            const = Q_info.quantize(-mean * coef)
        output = Q_info.quantize( Q_info.quantize(inputs * coef) + const)
        return output


###########################################
### Reimplemented Depthwise Convolution ###
###########################################
# parallel_iterations and swap_memory in tf.while_loops can be adjusted
def QuantizedDepthwiseConv2DCore(inputs, kernel, strides, rate, padding, data_format, Q_info):
    ''' Reimplementation of the 2D depthwise convolution layer.
    Args: 
        inputs:  [batch_size, image_height, image_width, input_channels] 
        kernel: [kernel_height, kernel_width, input_channels, output_channels]
    '''

    inputs, tf_data_format = _preprocess_conv2d_input(inputs, data_format)
    if tf_data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(tf_data_format))
        
    padding = _preprocess_padding(padding)
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value
    
    
    if not tf_while_loop:
        # work around of tf.slice bug in multi gpu condition
        if batch_size is None:
            batch_size=tf.shape(inputs)[:1]
        
        # work around of tf.slice bug in multi gpu condition
        if not isinstance(batch_size,int):
            batch_size=batch_size[0]
    
        # prepare kernel
        kernel_shape = kernel.get_shape()
    
        # get patch shape, needed for ofmap shape estimation
        output = tf.extract_image_patches(inputs, 
                                          ksizes=(1,kernel_shape.dims[0], kernel_shape.dims[1],1), 
                                          strides=strides,
                                          rates=rate,#[1,1,1,1],
                                          padding=padding )
        patch_shape = output.get_shape()
        #[batch, ofmap height, ofmap width, num of kernel psum * channel]
                
        kernel_tmp = tf.reshape(kernel, [1,1,1,patch_shape.dims[3].value])
        kernel_tmp = tf.tile(kernel_tmp,[batch_size,patch_shape.dims[1].value,patch_shape.dims[2].value,1])  
        #[batch, ofmap height, ofmap width, num of kernel psum * channel]
    
        output = tf.multiply(output, kernel_tmp)
        # quantize after multiplication
        output = Q_info.quantize(output)     
        
        output = tf.reshape(output, [batch_size,patch_shape.dims[1].value,patch_shape.dims[2].value,tf.reduce_prod(kernel_shape[0:2]),kernel_shape.dims[2].value])
        
        output = tf.reduce_sum(output,axis=3,keepdims=False)
        # quantize after accumulation
        output = Q_info.quantize(output)     

    else:
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
            out_tmp = Q_info.quantize(out_tmp)    
            
            out_tmp = tf.reshape(out_tmp, [1,patch_shape.dims[1].value,patch_shape.dims[2].value,tf.reduce_prod(kernel_shape[0:2]),kernel_shape.dims[2].value])
            
            out_tmp = tf.reduce_sum(out_tmp,axis=3,keepdims=False)
            # quantize after accumulation
            out_tmp = Q_info.quantize(out_tmp)     
                    
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

    return output


###############################
### Distributed Convolution ###
###############################
# Distributed convolution for evaluattion of partial sum in hardware accelerator 
# where its input feature map channel is too many for input buffer. Divide the convolution
# into several parallel convolutions to view the partial sum value and inject fault.
    
def DistributedConv2D(x, kernel, splits, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """Distributed 2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        splits: Either a 0-D integer `Tensor` indicating the number of splits 
            along split_dim or a 1-D integer `Tensor` containing the sizes of 
            each output tensor along split_dim. If a scalar then it must evenly
            divide `value.shape[axis]`; otherwise the sum of sizes along the 
            split dimension must match that of the `value`.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    if tf_data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(tf_data_format))

    padding = _preprocess_padding(padding)
    
    x=tf.split(x,splits,axis=3)
    kernel=tf.split(kernel,splits,axis=2)
    
    for i in range(len(x)):
        x[i] = tf.nn.convolution(
                input=x[i],
                filter=kernel[i],
                dilation_rate=dilation_rate,
                strides=strides,
                padding=padding,
                data_format=tf_data_format)
    return x

def QuantizedDistributedConv2DCore(x, kernel, splits, strides, dilation_rate, padding, data_format, Q_info):
    """2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        splits: Either a 0-D integer `Tensor` indicating the number of splits 
            along split_dim or a 1-D integer `Tensor` containing the sizes of 
            each output tensor along split_dim. If a scalar then it must evenly
            divide `value.shape[axis]`; otherwise the sum of sizes along the 
            split dimension must match that of the `value`.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    if tf_data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(tf_data_format))

    padding = _preprocess_padding(padding)
    
    x=tf.split(x,splits,axis=3)
    kernel=tf.split(kernel,splits,axis=2)
    
    for i in range(len(x)):
        x[i] = QuantizedConv2DCore(
                inputs=x[i],
                kernel=kernel[i],
                strides=strides,
                rate=dilation_rate,
                padding=padding,
                data_format=tf_data_format,
                Q_info=Q_info)
    return x


