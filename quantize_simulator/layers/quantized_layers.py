# -*- coding: utf-8 -*-

'''
reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-Tensorflow

@author: Yung-Yu Tsai

'''

import numpy as np

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Conv2D, BatchNormalization, DepthwiseConv2D
from keras import constraints
from keras import initializers

from layers.quantized_ops import quantize, clip_through
from testing.fault_ops import inject_layer_sa_fault_tensor


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
    ''' Quantized Dense layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, units, H=1., nb=16, fb=8, rounding_method='nearest', kernel_lr_multiplier='Glorot', bias_lr_multiplier=None,
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None], **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.nb = nb
        self.fb = fb
        self.rounding_method = rounding_method
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.weight_sa_fault_injection=weight_sa_fault_injection
        self.ifmap_sa_fault_injection=ifmap_sa_fault_injection
        self.ofmap_sa_fault_injection=ofmap_sa_fault_injection
        super(QuantizedDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        
        if self.weight_sa_fault_injection[0] is not None:
            quantized_kernel = inject_layer_sa_fault_tensor(quantized_kernel, self.weight_sa_fault_injection[0], self.nb, self.fb, rounding=self.rounding_method)
            
        inputs = quantize(inputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        
        if self.ifmap_sa_fault_injection is not None:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, self.nb, self.fb, rounding=self.rounding_method)
        
        output = K.dot(inputs, quantized_kernel)
        output = quantize(output, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        if self.use_bias:
            quantized_bias = quantize(self.bias, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            
            if self.weight_sa_fault_injection[1] is not None:
                quantized_bias = inject_layer_sa_fault_tensor(self.bias, self.weight_sa_fault_injection[1], self.nb, self.fb, rounding=self.rounding_method)

            output = K.bias_add(output, quantized_bias)
        if self.activation is not None:
            output = quantize(output, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            output = self.activation(output)
            
        if self.ofmap_sa_fault_injection is not None:
            output = inject_layer_sa_fault_tensor(output, self.ofmap_sa_fault_injection, self.nb, self.fb, rounding=self.rounding_method)



        return output
        
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier,
                  'nb': self.nb,
                  'fb': self.fb,
                  'rounding_method': self.rounding_method
                  }
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedConv2D(Conv2D):
    '''Quantized Convolution2D layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters, kernel_regularizer=None,activity_regularizer=None, kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None, H=1., nb=16, fb=8, rounding_method='nearest',
                 ifmap_sa_fault_injection=None, ofmap_sa_fault_injection=None, weight_sa_fault_injection=[None, None],**kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.nb = nb
        self.fb = fb
        self.rounding_method = rounding_method
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.activity_regularizer =activity_regularizer
        self.kernel_regularizer = kernel_regularizer
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
            
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        #self.bias_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        
        if self.weight_sa_fault_injection[0] is not None:
            quantized_kernel = inject_layer_sa_fault_tensor(quantized_kernel, self.weight_sa_fault_injection[0], self.nb, self.fb, rounding=self.rounding_method)

        
        inputs = quantize(inputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        
        if self.ifmap_sa_fault_injection is not None:
            inputs = inject_layer_sa_fault_tensor(inputs, self.ifmap_sa_fault_injection, self.nb, self.fb, rounding=self.rounding_method)


        inverse_kernel_lr_multiplier = 1./self.kernel_lr_multiplier
        inputs_qnn_gradient = (inputs - (1. - 1./inverse_kernel_lr_multiplier) * K.stop_gradient(inputs))\
                  * inverse_kernel_lr_multiplier

        outputs_qnn_gradient = K.conv2d(
            inputs_qnn_gradient,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        
        outputs_qnn_gradient = quantize(outputs_qnn_gradient, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)

        outputs = (outputs_qnn_gradient - (1. - 1./self.kernel_lr_multiplier) * K.stop_gradient(outputs_qnn_gradient))\
                  * self.kernel_lr_multiplier


        #outputs = outputs*K.mean(K.abs(self.kernel))

        if self.use_bias:
            quantized_bias = quantize(self.bias, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            
            if self.weight_sa_fault_injection[1] is not None:
                quantized_bias = inject_layer_sa_fault_tensor(self.bias, self.weight_sa_fault_injection[1], self.nb, self.fb, rounding=self.rounding_method)

            
            outputs = K.bias_add(
                outputs,
                quantized_bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs = quantize(outputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            return self.activation(outputs)
        
        if self.ofmap_sa_fault_injection is not None:
            outputs = inject_layer_sa_fault_tensor(outputs, self.ofmap_sa_fault_injection, self.nb, self.fb, rounding=self.rounding_method)


        return outputs
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier,
                  'nb': self.nb,
                  'fb': self.fb,
                  'rounding_method': self.rounding_method
                  }
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

QuantizedConvolution2D = QuantizedConv2D


class QuantizedBatchNormalization(BatchNormalization):
    ''' Quantized BatchNormalization layer
    References: 
    "Pytorch Playground: Base pretrained models and datasets in pytorch." [https://github.com/aaron-xichen/pytorch-playground]
    '''
    def __init__(self,
                 H=1, nb=16, fb=8, rounding_method='nearest', **kwargs):
        super(QuantizedBatchNormalization, self).__init__(**kwargs)
        self.H = H
        self.nb = nb
        self.fb = fb
        self.rounding_method = rounding_method

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
                    
                broadcast_moving_mean = quantize(broadcast_moving_mean, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                broadcast_moving_variance = quantize(broadcast_moving_variance, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                broadcast_beta = quantize(broadcast_beta, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                broadcast_gamma = quantize(broadcast_gamma, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                    
                return K.batch_normalization(
                    quantize(inputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method),
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                moving_mean = quantize(self.moving_mean, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                moving_variance = quantize(self.moving_variance, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                beta = quantize(self.beta, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                gamma = quantize(self.gamma, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
                
                return K.batch_normalization(
                    quantize(inputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method),
                    moving_mean,
                    moving_variance,
                    beta,
                    gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return quantize(normalize_inference(), nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)

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
        config = {'H': self.H,
                  'nb': self.nb,
                  'fb': self.fb,
                  'rounding_method': self.rounding_method
                  }
        base_config = super(QuantizedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class QuantizedDepthwiseConv2D(DepthwiseConv2D):
    '''Quantized DepthwiseConv2D layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self,
                 kernel_size,
                 H=1.,
                 nb=16,
                 fb=8,
                 rounding_method='nearest',
                 **kwargs):
        super(QuantizedDepthwiseConv2D, self).__init__(kernel_size, **kwargs)
        self.H = H
        self.nb = nb
        self.fb = fb
        self.rounding_method = rounding_method

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
        inputs=quantize(inputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        quantized_depthwise_kernel=quantize(self.depthwise_kernel, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
        
        outputs = K.depthwise_conv2d(
            inputs,
            quantized_depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)
                
        if self.bias:
            outputs = quantize(outputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            quantized_bias = quantize(self.bias, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            outputs = K.bias_add(
                outputs,
                quantized_bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs=quantize(outputs, nb=self.nb, fb=self.fb, rounding_method=self.rounding_method)
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = {'H': self.H,
                  'nb': self.nb,
                  'fb': self.fb,
                  'rounding_method': self.rounding_method
                  }
        base_config = super(QuantizedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
