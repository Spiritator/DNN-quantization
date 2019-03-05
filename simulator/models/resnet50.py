"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras_applications import  get_keras_submodule

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')

from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions, _obtain_input_shape

from layers.quantized_layers import QuantizedConv2D, QuantizedDense, QuantizedBatchNormalization, QuantizedFlatten

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def identity_block(input_tensor, kernel_size, filters, stage, block, 
                   nbits=24, fbits=9, BN_nbits=24, BN_fbits=9, rounding_method='nearest',quant_mode='hybrid'):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    print('building stage %d block %s ...'%(stage,block))

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1),
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2a',
                        quant_mode=quant_mode)(input_tensor)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2a',
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size,
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        padding='same', 
                        name=conv_name_base + '2b',
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2b',
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2c',
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2c',
                                    quant_mode=quant_mode)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               nbits=24, fbits=9, BN_nbits=24, BN_fbits=9, rounding_method='nearest',quant_mode='hybrid'):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    print('building stage %d block %s ...'%(stage,block))

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1), 
                        strides=strides,
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2a',
                        quant_mode=quant_mode)(input_tensor)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2a',
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size, 
                        padding='same',
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2b',
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2b',
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2c',
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2c',
                                    quant_mode=quant_mode)(x)

    shortcut = QuantizedConv2D(filters3, 
                               kernel_size=(1, 1), 
                               strides=strides,
                               nb=nbits,
                               fb=fbits, 
                               rounding_method=rounding_method,
                               name=conv_name_base + '1',
                               quant_mode=quant_mode)(input_tensor)
    shortcut = QuantizedBatchNormalization(nb=BN_nbits,
                                           fb=BN_fbits,
                                           rounding_method=rounding_method,
                                           axis=bn_axis, 
                                           name=bn_name_base + '1',
                                           quant_mode=quant_mode)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def QuantizedResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             batch_size=None,
             nbits=24,
             fbits=9, 
             BN_nbits=None, 
             BN_fbits=None,
             rounding_method='nearest',
             quant_mode='hybrid'):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    print('Building model : Quantized ResNet50')
    
    if BN_nbits is None:
        BN_nbits=nbits

    if BN_fbits is None:
        BN_fbits=fbits
    
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, batch_shape=(batch_size,)+input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape, batch_shape=(batch_size,)+input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    print('building stage 1 ...')

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = QuantizedConv2D(64, 
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='valid',
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name='conv1',
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method,
                                    axis=bn_axis, 
                                    name='bn_conv1',
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    print('building stage 2 ...')

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 3 ...')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 4 ...')
    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 5 ...')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', nbits=nbits, fbits=fbits, BN_nbits=BN_nbits, BN_fbits=BN_fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = QuantizedFlatten()(x)
        x = QuantizedDense(classes, activation='softmax', name='fc1000',
                           nb=nbits,
                           fb=fbits, 
                           rounding_method=rounding_method,
                           quant_mode=quant_mode)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model



#==============================================================================
#    FUSED BATCHNORMALIZATION MODEL
#==============================================================================

def identity_block_fused_BN(input_tensor, kernel_size, filters, stage, block, 
                            nbits=24, fbits=9, rounding_method='nearest',quant_mode='hybrid'):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    
    print('building stage %d block %s ...'%(stage,block))

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1),
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2a',
                        quant_mode=quant_mode)(input_tensor)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size,
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        padding='same', 
                        name=conv_name_base + '2b',
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2c',
                        quant_mode=quant_mode)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_fused_BN(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               nbits=24, fbits=9, rounding_method='nearest',quant_mode='hybrid'):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    
    print('building stage %d block %s ...'%(stage,block))

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1), 
                        strides=strides,
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2a',
                        quant_mode=quant_mode)(input_tensor)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size, 
                        padding='same',
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2b',
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name=conv_name_base + '2c',
                        quant_mode=quant_mode)(x)

    shortcut = QuantizedConv2D(filters3, 
                               kernel_size=(1, 1), 
                               strides=strides,
                               nb=nbits,
                               fb=fbits, 
                               rounding_method=rounding_method,
                               name=conv_name_base + '1',
                               quant_mode=quant_mode)(input_tensor)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def QuantizedResNet50FusedBN(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             batch_size=None,
             nbits=24,
             fbits=9, 
             rounding_method='nearest',
             quant_mode='hybrid'):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    print('Building model : Quantized ResNet50')
    
    
    if not os.path.exists(weights):
        raise ValueError('The `weights` argument must be the path to the weights file to be loaded. File not found!')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, batch_shape=(batch_size,)+input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape, batch_shape=(batch_size,)+input_shape)
        else:
            img_input = input_tensor
        
    print('building stage 1 ...')

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = QuantizedConv2D(64, 
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='valid',
                        nb=nbits,
                        fb=fbits, 
                        rounding_method=rounding_method,
                        name='conv1',
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    print('building stage 2 ...')

    x = conv_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='b', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='c', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 3 ...')

    x = conv_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='a', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='b', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='c', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='d', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 4 ...')
    
    x = conv_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='a', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='b', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='c', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='d', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='e', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='f', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    print('building stage 5 ...')

    x = conv_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='a', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='b', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)
    x = identity_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='c', nbits=nbits, fbits=fbits, rounding_method=rounding_method, quant_mode=quant_mode)

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = QuantizedFlatten()(x)
        x = QuantizedDense(classes, activation='softmax', name='fc1000',
                           nb=nbits,
                           fb=fbits, 
                           rounding_method=rounding_method,
                           quant_mode=quant_mode)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # load weights
    if weights is not None:
        model.load_weights(weights)

    return model


