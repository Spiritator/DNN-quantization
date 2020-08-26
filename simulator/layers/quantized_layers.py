# -*- coding: utf-8 -*-

'''
The custom quantized layers of Keras layer module. Support fault injection to input, weight, output of layer.

@author: Yung-Yu Tsai

'''

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.layers import InputSpec, Layer, Dense, Conv2D, BatchNormalization, DepthwiseConv2D, Flatten
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.python.keras.utils import conv_utils

from .quantized_ops import quantizer
from ..testing.fault_ops import inject_layer_sa_fault_tensor
from .intra_layer_ops import QuantizedDenseCore, QuantizedConv2DCore, QuantizedBatchNormalizationCore, QuantizedDepthwiseConv2DCore, DistributedConv2D, QuantizedDistributedConv2DCore


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        #todo: switch for clip through?
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"name": self.__call__.__name__,
                "min_value": self.min_value,
                "max_value": self.max_value}


class QuantizedDense(Dense):
    ''' Quantized Dense layer '''
    def __init__(self, units, quantizers, quant_mode='hybrid',
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None],
                 mac_unit=None, last_layer=False, **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.quantizer=quantizers
        self.quant_mode = quant_mode
        self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection
        self.mac_unit=mac_unit
        self.last_layer=last_layer
        super(QuantizedDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]
            
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        if self.quant_mode not in [None,'extrinsic','hybrid','intrinsic']:
            raise ValueError('Invalid quantization mode. The \'quant_mode\' augment must be one of \'extrinsic\' , \'intrinsic\' , \'hybrid\' or None.')
        
        if isinstance(self.quantizer,list) and len(self.quantizer)==3:
            quantizer_input =self.quantizer[0]
            quantizer_weight =self.quantizer[1]
            quantizer_output =self.quantizer[2]
        else:
            quantizer_input =self.quantizer
            quantizer_weight =self.quantizer
            quantizer_output =self.quantizer
            
        
        if self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = quantizer_weight.quantize(self.kernel)
        
        if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = inject_layer_sa_fault_tensor(quantized_kernel, self.weight_sa_fault_injection[0], quantizer_weight)
        
        if self.quant_mode in ['hybrid','intrinsic']:
            inputs = quantizer_input.quantize(inputs)
        
        if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, quantizer_input)
        
        if self.quant_mode == 'intrinsic':
            output = QuantizedDenseCore(inputs, quantized_kernel, quantizer_output)
        elif self.quant_mode == 'hybrid':
            output = K.dot(inputs, quantized_kernel)
            output = quantizer_output.quantize(output)                        
        elif self.quant_mode in ['extrinsic',None]:
            output = K.dot(inputs, self.kernel)
            
        if self.use_bias:
            if self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = quantizer_weight.quantize(self.bias)
            
            if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = inject_layer_sa_fault_tensor(quantized_bias, self.weight_sa_fault_injection[1], quantizer_weight)
                
            if self.quant_mode in ['hybrid','intrinsic']:
                output = K.bias_add(output, quantized_bias)
                output = quantizer_output.quantize(output)
            elif self.quant_mode in ['extrinsic',None]:
                output = K.bias_add(output, self.bias)
            

            
        if self.activation is not None:
            output = self.activation(output)
        
        if self.quant_mode in ['extrinsic','hybrid','intrinsic'] and not self.last_layer:
            output = quantizer_output.quantize(output)
            
        if self.ofmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic'] and not self.last_layer:
            if self.mac_unit is None:
                output = inject_layer_sa_fault_tensor(output, self.ofmap_sa_fault_injection, quantizer_output)
            else:
                output = self.mac_unit.inject_mac_math_fault_tensor(inputs, quantized_kernel, output, 
                                                                    self.ofmap_sa_fault_injection,
                                                                    layer_type='Dense')




        return output
        
        
    def get_config(self):
        if isinstance(self.quantizer,list):
            nb=[quant.nb for quant in self.quantizer]
            fb=[quant.fb for quant in self.quantizer]
            rounding_method=[quant.rounding_method for quant in self.quantizer]
        else:
            nb=self.quantizer.nb
            fb=self.quantizer.fb
            rounding_method=self.quantizer.rounding_method
        config = {'quant_mode': self.quant_mode,
                  'nb': nb,
                  'fb': fb,
                  'rounding_method': rounding_method
                  }
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedConv2D(Conv2D):
    '''Quantized Convolution2D layer '''
    def __init__(self, filters, quantizers, quant_mode='hybrid',
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None],
                 mac_unit=None, last_layer=False, **kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        self.quantizer=quantizers
        self.quant_mode = quant_mode
        self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection
        self.mac_unit=mac_unit
        self.last_layer=last_layer
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.quant_mode not in [None,'extrinsic','hybrid','intrinsic']:
            raise ValueError('Invalid quantization mode. The \'quant_mode\' augment must be one of \'extrinsic\' , \'intrinsic\' , \'hybrid\' or None.')

        if isinstance(self.quantizer,list) and len(self.quantizer)==3:
            quantizer_input =self.quantizer[0]
            quantizer_weight =self.quantizer[1]
            quantizer_output =self.quantizer[2]
        else:
            quantizer_input =self.quantizer
            quantizer_weight =self.quantizer
            quantizer_output =self.quantizer

        
        if self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = quantizer_weight.quantize(self.kernel)
        
        if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = inject_layer_sa_fault_tensor(quantized_kernel, self.weight_sa_fault_injection[0], quantizer_weight)

        if self.quant_mode in ['hybrid','intrinsic']:
            inputs = quantizer_input.quantize(inputs)
        
        if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, quantizer_input)


        if self.quant_mode == 'intrinsic':
            strides = (1,self.strides[0],self.strides[1],1)
            dilation_rate = (1,self.dilation_rate[0],self.dilation_rate[1],1)
            outputs = QuantizedConv2DCore(
                    inputs,
                    quantized_kernel,
                    strides, dilation_rate,
                    self.padding,
                    self.data_format,
                    quantizer_output)
        elif self.quant_mode == 'hybrid':
            outputs = K.conv2d(
                    inputs,
                    quantized_kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)
            outputs = quantizer_output.quantize(outputs)                        
        elif self.quant_mode in ['extrinsic',None]:
            outputs = K.conv2d(
                    inputs,
                    self.kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)
            

        if self.use_bias:
            if self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = quantizer_weight.quantize(self.bias)
            
            if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = inject_layer_sa_fault_tensor(quantized_bias, self.weight_sa_fault_injection[1], quantizer_weight)

            if self.quant_mode in ['hybrid','intrinsic']:
                outputs = K.bias_add(
                        outputs,
                        quantized_bias,
                        data_format=self.data_format)          
                outputs = quantizer_output.quantize(outputs)
            elif self.quant_mode in ['extrinsic',None]:
                outputs = K.bias_add(
                        outputs,
                        self.bias,
                        data_format=self.data_format)  


        if self.activation is not None:
            outputs = self.activation(outputs)
        
        if self.quant_mode in ['extrinsic','hybrid','intrinsic'] and not self.last_layer:
            outputs = quantizer_output.quantize(outputs)
        
        if self.ofmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic'] and not self.last_layer:
            if self.mac_unit is None:
                outputs = inject_layer_sa_fault_tensor(outputs, self.ofmap_sa_fault_injection, quantizer_output)
            else:
                outputs = self.mac_unit.inject_mac_math_fault_tensor(inputs, quantized_kernel, outputs, 
                                                                     self.ofmap_sa_fault_injection,
                                                                     layer_type='Conv2D',
                                                                     ksizes=self.kernel_size, 
                                                                     padding=self.padding, 
                                                                     dilation_rates=self.dilation_rate)

        return outputs
        
    def get_config(self):
        if isinstance(self.quantizer,list):
            nb=[quant.nb for quant in self.quantizer]
            fb=[quant.fb for quant in self.quantizer]
            rounding_method=[quant.rounding_method for quant in self.quantizer]
        else:
            nb=self.quantizer.nb
            fb=self.quantizer.fb
            rounding_method=self.quantizer.rounding_method
        config = {'quant_mode': self.quant_mode,
                  'nb': nb,
                  'fb': fb,
                  'rounding_method': rounding_method
                  }
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

QuantizedConvolution2D = QuantizedConv2D


class QuantizedBatchNormalization(BatchNormalization):
    ''' Quantized BatchNormalization layer '''
    def __init__(self, quantizers, quant_mode='hybrid',
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None, None, None],**kwargs):
        super(QuantizedBatchNormalization, self).__init__(**kwargs)
        self.quantizer=quantizers
        self.quant_mode = quant_mode
        if (isinstance(weight_sa_fault_injection,list) and len(weight_sa_fault_injection)==2) or weight_sa_fault_injection is None:
            self.weight_sa_fault_injection=[None, None, None, None]
        else:            
            self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection


    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        if self.quant_mode not in [None,'extrinsic','hybrid','intrinsic']:
            raise ValueError('Invalid quantization mode. The \'quant_mode\' augment must be one of \'extrinsic\' , \'intrinsic\' , \'hybrid\' or None.')

        if isinstance(self.quantizer,list) and len(self.quantizer)==3:
            quantizer_input =self.quantizer[0]
            quantizer_weight =self.quantizer[1]
            quantizer_output =self.quantizer[2]
        else:
            quantizer_input =self.quantizer
            quantizer_weight =self.quantizer
            quantizer_output =self.quantizer

        
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                    
                if self.quant_mode in ['hybrid','intrinsic']:
                    broadcast_moving_mean = quantizer_weight.quantize(broadcast_moving_mean)
                    broadcast_moving_variance = quantizer_weight.quantize(broadcast_moving_variance)
                    if self.center:
                        broadcast_beta = quantizer_weight.quantize(broadcast_beta)
                    if self.scale:
                        broadcast_gamma = quantizer_weight.quantize(broadcast_gamma)
                        
                if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.scale:
                    broadcast_gamma = inject_layer_sa_fault_tensor(broadcast_gamma, self.weight_sa_fault_injection[0], quantizer_weight)
                    
                if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.center:
                    broadcast_beta = inject_layer_sa_fault_tensor(broadcast_beta, self.weight_sa_fault_injection[1], quantizer_weight)
                    
                if self.weight_sa_fault_injection[2] is not None and self.quant_mode in ['hybrid','intrinsic']:
                    broadcast_moving_mean = inject_layer_sa_fault_tensor(broadcast_moving_mean, self.weight_sa_fault_injection[2], quantizer_weight)
                    
                if self.weight_sa_fault_injection[3] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.scale:
                    broadcast_moving_variance = inject_layer_sa_fault_tensor(broadcast_moving_variance, self.weight_sa_fault_injection[3], quantizer_weight)
                    
                    
                if self.quant_mode in ['hybrid','intrinsic']:
                    quantized_inputs = quantizer_input.quantize(inputs)
                    
                if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
                    quantized_inputs = inject_layer_sa_fault_tensor(quantized_inputs, self.ifmap_sa_fault_injection, quantizer_input)

                
                if self.quant_mode == 'intrinsic':
                    return QuantizedBatchNormalizationCore(
                            quantized_inputs,
                            broadcast_moving_mean,
                            broadcast_moving_variance,
                            broadcast_beta,
                            broadcast_gamma,
                            self.epsilon,
                            quantizer_output)
                elif self.quant_mode == 'hybrid':
                    output=K.batch_normalization(
                            quantized_inputs,
                            broadcast_moving_mean,
                            broadcast_moving_variance,
                            broadcast_beta,
                            broadcast_gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)
                    return quantizer_output.quantize(output)     
                elif self.quant_mode == 'extrinsic':
                    output=K.batch_normalization(
                            inputs,
                            broadcast_moving_mean,
                            broadcast_moving_variance,
                            broadcast_beta,
                            broadcast_gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)
                    return quantizer_output.quantize(output)
                elif self.quant_mode is None:
                    return K.batch_normalization(
                            inputs,
                            broadcast_moving_mean,
                            broadcast_moving_variance,
                            broadcast_beta,
                            broadcast_gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)
                    
            else:
                if self.quant_mode in ['hybrid','intrinsic']:
                    moving_mean = quantizer_weight.quantize(self.moving_mean)
                    moving_variance = quantizer_weight.quantize(self.moving_variance)
                    if self.center:
                        beta = quantizer_weight.quantize(self.beta)
                    else:
                        beta = self.beta
                    if self.scale:
                        gamma = quantizer_weight.quantize(self.gamma)
                    else:
                        gamma = self.gamma
                        
                if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.scale:
                    gamma = inject_layer_sa_fault_tensor(gamma, self.weight_sa_fault_injection[0], quantizer_weight)
                    
                if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.center:
                    beta = inject_layer_sa_fault_tensor(beta, self.weight_sa_fault_injection[1], quantizer_weight)
                    
                if self.weight_sa_fault_injection[2] is not None and self.quant_mode in ['hybrid','intrinsic']:
                    moving_mean = inject_layer_sa_fault_tensor(moving_mean, self.weight_sa_fault_injection[2], quantizer_weight)
                    
                if self.weight_sa_fault_injection[3] is not None and self.quant_mode in ['hybrid','intrinsic'] and self.scale:
                    moving_variance = inject_layer_sa_fault_tensor(moving_variance, self.weight_sa_fault_injection[3], quantizer_weight)
                    
                    
                if self.quant_mode in ['hybrid','intrinsic']:
                    quantized_inputs = quantizer_input.quantize(inputs)
                    
                if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
                    quantized_inputs = inject_layer_sa_fault_tensor(quantized_inputs, self.ifmap_sa_fault_injection, quantizer_input)

                
                if self.quant_mode == 'intrinsic':
                    return QuantizedBatchNormalizationCore(
                            quantized_inputs,
                            moving_mean,
                            moving_variance,
                            beta,
                            gamma,
                            self.epsilon,
                            quantizer_output)
                elif self.quant_mode == 'hybrid':
                    output=K.batch_normalization(
                            quantized_inputs,
                            moving_mean,
                            moving_variance,
                            beta,
                            gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)
                    return quantizer_output.quantize(output)     
                elif self.quant_mode == 'extrinsic':
                    output=K.batch_normalization(
                            inputs,
                            self.moving_mean,
                            self.moving_variance,
                            self.beta,
                            self.gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)
                    return quantizer_output.quantize(output)
                elif self.quant_mode == None:
                    return K.batch_normalization(
                            inputs,
                            self.moving_mean,
                            self.moving_variance,
                            self.beta,
                            self.gamma,
                            axis=self.axis,
                            epsilon=self.epsilon)


                

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            if self.ofmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
                return inject_layer_sa_fault_tensor(normalize_inference(), self.ofmap_sa_fault_injection, quantizer_output)
            else:
                return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        if isinstance(self.quantizer,list):
            nb=[quant.nb for quant in self.quantizer]
            fb=[quant.fb for quant in self.quantizer]
            rounding_method=[quant.rounding_method for quant in self.quantizer]
        else:
            nb=self.quantizer.nb
            fb=self.quantizer.fb
            rounding_method=self.quantizer.rounding_method
        config = {'quant_mode': self.quant_mode,
                  'nb': nb,
                  'fb': fb,
                  'rounding_method': rounding_method
                  }
        base_config = super(QuantizedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class QuantizedDepthwiseConv2D(DepthwiseConv2D):
    '''Quantized DepthwiseConv2D layer '''
    def __init__(self,
                 kernel_size,
                 quantizers,
                 quant_mode='hybrid',
                 ifmap_sa_fault_injection=None, 
                 ofmap_sa_fault_injection=None, 
                 weight_sa_fault_injection=[None, None],
                 mac_unit=None, 
                 last_layer=False,
                 **kwargs):
        super(QuantizedDepthwiseConv2D, self).__init__(kernel_size, **kwargs)
        self.quantizer=quantizers
        self.quant_mode = quant_mode
        self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection
        self.mac_unit=mac_unit
        self.last_layer=last_layer

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        if isinstance(self.quantizer,list) and len(self.quantizer)==3:
            quantizer_input =self.quantizer[0]
            quantizer_weight =self.quantizer[1]
            quantizer_output =self.quantizer[2]
        else:
            quantizer_input =self.quantizer
            quantizer_weight =self.quantizer
            quantizer_output =self.quantizer

        
        if self.quant_mode in ['hybrid','intrinsic']:
            inputs = quantizer_input.quantize(inputs)
        
        if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, quantizer_input)

        if self.quant_mode in ['hybrid','intrinsic']:
            quantized_depthwise_kernel=quantizer_weight.quantize(self.depthwise_kernel)
        
        if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic']:
            quantized_depthwise_kernel= inject_layer_sa_fault_tensor(quantized_depthwise_kernel, self.weight_sa_fault_injection[0], quantizer_weight)

        if self.quant_mode == 'intrinsic':
            strides = (1,self.strides[0],self.strides[1],1)
            dilation_rate = (1,self.dilation_rate[0],self.dilation_rate[1],1)
            outputs = QuantizedDepthwiseConv2DCore(
                    inputs,
                    quantized_depthwise_kernel,
                    strides, dilation_rate,
                    self.padding,
                    self.data_format,
                    quantizer_output)
        elif self.quant_mode == 'hybrid':
            outputs = K.depthwise_conv2d(
                    inputs,
                    quantized_depthwise_kernel,
                    strides=self.strides,
                    padding=self.padding,
                    dilation_rate=self.dilation_rate,
                    data_format=self.data_format)
            outputs = quantizer_output.quantize(outputs)
        elif self.quant_mode in ['extrinsic',None]:
            outputs = K.depthwise_conv2d(
                    inputs,
                    self.depthwise_kernel,
                    strides=self.strides,
                    padding=self.padding,
                    dilation_rate=self.dilation_rate,
                    data_format=self.data_format)

        
                
        if self.bias:
            if self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = quantizer_weight.quantize(self.bias)
            
            if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = inject_layer_sa_fault_tensor(quantized_bias, self.weight_sa_fault_injection[1], quantizer_weight)

            if self.quant_mode in ['hybrid','intrinsic']:
                outputs = K.bias_add(
                    outputs,
                    quantized_bias,
                    data_format=self.data_format)
                outputs = quantizer_output.quantize(outputs)
            elif self.quant_mode in ['extrinsic',None]:
                outputs = K.bias_add(
                        outputs,
                        self.bias,
                        data_format=self.data_format)  


        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.quant_mode in ['extrinsic','hybrid','intrinsic']:
            outputs = quantizer_output.quantize(outputs)
        
        if self.ofmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic'] and not self.last_layer:
            if self.mac_unit is None:
                outputs = inject_layer_sa_fault_tensor(outputs, self.ofmap_sa_fault_injection, quantizer_output)
            else:
                outputs = self.mac_unit.inject_mac_math_fault_tensor(inputs, quantized_depthwise_kernel, outputs, 
                                                                     self.ofmap_sa_fault_injection,
                                                                     layer_type='DepthwiseConv2D',
                                                                     ksizes=self.kernel_size, 
                                                                     padding=self.padding, 
                                                                     dilation_rates=self.dilation_rate)

        return outputs

    def get_config(self):
        if isinstance(self.quantizer,list):
            nb=[quant.nb for quant in self.quantizer]
            fb=[quant.fb for quant in self.quantizer]
            rounding_method=[quant.rounding_method for quant in self.quantizer]
        else:
            nb=self.quantizer.nb
            fb=self.quantizer.fb
            rounding_method=self.quantizer.rounding_method
        config = {'quant_mode': self.quant_mode,
                  'nb': nb,
                  'fb': fb,
                  'rounding_method': rounding_method
                  }
        base_config = super(QuantizedDepthwiseConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedFlatten(Flatten):
    '''
    Work around bug of not showing shape of flatten and reshape layer output in keras.
    Custom remake a Flatten layer for the reliability analysis and quant_mode operation after flatten layer.
    '''
    def __init__(self, **kwargs):
        super(QuantizedFlatten, self).__init__(**kwargs)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)
        
        if inputs.shape.dims[0].value is None:
            #return K.batch_flatten(inputs)
            return tf.reshape(inputs, [-1,np.prod(inputs.shape.dims[1:])])
        else:
            return tf.reshape(inputs, [inputs.shape.dims[0].value,-1])

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(QuantizedFlatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class QuantizedDistributedConv2D(Conv2D):
    '''Quantized Distributed Convolution2D layer'''
    # Distributed convolution for evaluattion of partial sum in hardware accelerator 
    # where its input feature map channel is too many for input buffer. Divide the convolution
    # into several parallel convolutions to view the partial sum value and inject fault.

    def __init__(self, filters, split_type, splits, quantizers, quant_mode='hybrid',
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None],**kwargs):
        super(QuantizedDistributedConv2D, self).__init__(filters, **kwargs)
        self.split_type = split_type
        self.splits = splits
        self.quantizer=quantizers
        self.quant_mode = quant_mode
        self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.quant_mode not in [None,'extrinsic','hybrid','intrinsic']:
            raise ValueError('Invalid quantization mode. The \'quant_mode\' augment must be one of \'extrinsic\' , \'intrinsic\' , \'hybrid\' or None.')

        if isinstance(self.quantizer,list) and len(self.quantizer)==3:
            quantizer_input =self.quantizer[0]
            quantizer_weight =self.quantizer[1]
            quantizer_output =self.quantizer[2]
        else:
            quantizer_input =self.quantizer
            quantizer_weight =self.quantizer
            quantizer_output =self.quantizer

        
        if self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = quantizer_weight.quantize(self.kernel)
        
        if self.weight_sa_fault_injection[0] is not None and self.quant_mode in ['hybrid','intrinsic']:
            quantized_kernel = inject_layer_sa_fault_tensor(quantized_kernel, self.weight_sa_fault_injection[0], quantizer_weight)

        if self.quant_mode in ['hybrid','intrinsic']:
            inputs = quantizer_input.quantize(inputs)
        
        if self.ifmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, quantizer_input)


        if self.quant_mode == 'intrinsic':
            strides = (1,self.strides[0],self.strides[1],1)
            dilation_rate = (1,self.dilation_rate[0],self.dilation_rate[1],1)
            outputs = QuantizedDistributedConv2DCore(
                    inputs,
                    quantized_kernel,
                    self.split_type,
                    self.splits,
                    strides, dilation_rate,
                    self.padding,
                    self.data_format,
                    quantizer_output)
        elif self.quant_mode == 'hybrid':
            outputs = DistributedConv2D(
                    inputs,
                    quantized_kernel,
                    split_type=self.split_type,
                    splits=self.splits,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)
            for i in range(len(outputs)):
                outputs[i] = quantizer_output.quantize(outputs[i])
        elif self.quant_mode in ['extrinsic',None]:
            outputs = DistributedConv2D(
                    inputs,
                    self.kernel,
                    split_type=self.split_type,
                    splits=self.splits,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)
            

        if self.use_bias:
            if self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = quantizer_weight.quantize(self.bias)
            
            if self.weight_sa_fault_injection[1] is not None and self.quant_mode in ['hybrid','intrinsic']:
                quantized_bias = inject_layer_sa_fault_tensor(quantized_bias, self.weight_sa_fault_injection[1], quantizer_weight)

            if self.quant_mode in ['hybrid','intrinsic']:
                outputs[0] = K.bias_add(
                        outputs[0],
                        quantized_bias,
                        data_format=self.data_format)          
                outputs[0] = quantizer_output.quantize(outputs[0])
            elif self.quant_mode in ['extrinsic',None]:
                outputs[0] = K.bias_add(
                        outputs[0],
                        self.bias,
                        data_format=self.data_format)  


        if self.activation is not None:
            for i in range(len(outputs)):
                outputs[i] = self.activation(outputs[i])
        
        if self.quant_mode in ['extrinsic','hybrid','intrinsic']:
            for i in range(len(outputs)):
                outputs[i] = quantizer_output.quantize(outputs[i])
        
        if self.ofmap_sa_fault_injection is not None and self.quant_mode in ['hybrid','intrinsic']:
            if not isinstance(self.ofmap_sa_fault_injection,list) or len(outputs) is not len(self.ofmap_sa_fault_injection):
                raise ValueError('The output has %d sub-group, but output fault list got %d item can\'t match.'%(len(outputs),len(self.ofmap_sa_fault_injection)))
                
            for i in range(len(outputs)):
                outputs[i] = inject_layer_sa_fault_tensor(outputs[i], self.ofmap_sa_fault_injection[i], quantizer_output)


        return outputs
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            if isinstance(self.splits,int):
                return [(input_shape[0],) + tuple(new_space) + (self.filters,) for i in range(self.splits)]
            elif isinstance(self.splits,list):
                n_splt=1
                for splt in self.splits:
                    if isinstance(splt,int):
                        n_splt=n_splt*splt
                    elif isinstance(splt,list):
                        n_splt=n_splt*len(splt)
                return [(input_shape[0],) + tuple(new_space) + (self.filters,) for i in range(n_splt)]
            else:
                raise ValueError('splits augment must be integer or list.')
                
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            if isinstance(self.splits,int):
                return [(input_shape[0],self.filters) + tuple(new_space) for i in range(self.splits)]
            elif isinstance(self.splits,list):
                n_splt=1
                for splt in self.splits:
                    if isinstance(splt,int):
                        n_splt=n_splt*splt
                    elif isinstance(splt,list):
                        n_splt=n_splt*len(splt)
                return [(input_shape[0],self.filters) + tuple(new_space) for i in range(n_splt)]
            else:
                raise ValueError('splits augment must be integer or list.')
        
    def get_config(self):
        if isinstance(self.quantizer,list):
            nb=[quant.nb for quant in self.quantizer]
            fb=[quant.fb for quant in self.quantizer]
            rounding_method=[quant.rounding_method for quant in self.quantizer]
        else:
            nb=self.quantizer.nb
            fb=self.quantizer.fb
            rounding_method=self.quantizer.rounding_method
        config = {'quant_mode': self.quant_mode,
                  'splits': self.splits,
                  'nb': nb,
                  'fb': fb,
                  'rounding_method': rounding_method
                  }
        base_config = super(QuantizedDistributedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


