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
import warnings

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

from tqdm import tqdm
from ..layers.quantized_layers import QuantizedConv2D, QuantizedDense, QuantizedBatchNormalization, QuantizedFlatten
from ..layers.quantized_ops import quantizer,build_layer_quantizer

def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format='channels_last', **kwargs)

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

import keras.backend as backend
import keras.layers as layers
import keras.models as models
import keras.utils as keras_utils



def identity_block(input_tensor, 
                   kernel_size, 
                   filters, 
                   stage, 
                   block, 
                   layer_quantizer, 
                   layer_BN_quantizer, 
                   quant_mode='hybrid',
                   ifmap_fault_dict_list=None, 
                   ofmap_fault_dict_list=None, 
                   weight_fault_dict_list=None,
                   mac_unit=None):
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
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(10)]
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(10)]
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(10)]
    

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1),
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2a',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[0],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[0],
                        weight_sa_fault_injection=weight_fault_dict_list[0],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(input_tensor)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2a',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[1],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[1],
                                    weight_sa_fault_injection=weight_fault_dict_list[1],
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size,
                        quantizers=layer_quantizer,
                        padding='same', 
                        name=conv_name_base + '2b',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[3],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[3],
                        weight_sa_fault_injection=weight_fault_dict_list[3],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2b',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[4],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[4],
                                    weight_sa_fault_injection=weight_fault_dict_list[4],
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2c',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[6],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[6],
                        weight_sa_fault_injection=weight_fault_dict_list[6],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2c',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[7],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[7],
                                    weight_sa_fault_injection=weight_fault_dict_list[7],
                                    quant_mode=quant_mode)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               layer_quantizer, 
               layer_BN_quantizer,
               strides=(2, 2),
               quant_mode='hybrid',
               ifmap_fault_dict_list=None, 
               ofmap_fault_dict_list=None, 
               weight_fault_dict_list=None,
               mac_unit=None):
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
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(12)]
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(12)]
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(12)]
    

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1), 
                        strides=strides,
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2a',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[0],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[0],
                        weight_sa_fault_injection=weight_fault_dict_list[0],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(input_tensor)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2a',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[1],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[1],
                                    weight_sa_fault_injection=weight_fault_dict_list[1],
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size, 
                        padding='same',
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2b',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[3],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[3],
                        weight_sa_fault_injection=weight_fault_dict_list[3],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2b',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[4],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[4],
                                    weight_sa_fault_injection=weight_fault_dict_list[4],
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2c',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[6],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[6],
                        weight_sa_fault_injection=weight_fault_dict_list[6],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name=bn_name_base + '2c',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[8],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[8],
                                    weight_sa_fault_injection=weight_fault_dict_list[8],
                                    quant_mode=quant_mode)(x)

    shortcut = QuantizedConv2D(filters3, 
                               kernel_size=(1, 1), 
                               strides=strides,
                               quantizers=layer_quantizer,
                               name=conv_name_base + '1',
                               ifmap_sa_fault_injection=ifmap_fault_dict_list[7],
                               ofmap_sa_fault_injection=ofmap_fault_dict_list[7],
                               weight_sa_fault_injection=weight_fault_dict_list[7],
                               mac_unit=mac_unit,
                               quant_mode=quant_mode)(input_tensor)
    shortcut = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                           axis=bn_axis, 
                                           name=bn_name_base + '1',
                                           ifmap_sa_fault_injection=ifmap_fault_dict_list[9],
                                           ofmap_sa_fault_injection=ofmap_fault_dict_list[9],
                                           weight_sa_fault_injection=weight_fault_dict_list[9],
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
             quant_mode='hybrid',
             ifmap_fault_dict_list=None, 
             ofmap_fault_dict_list=None, 
             weight_fault_dict_list=None,
             mac_unit=None,
             overflow_mode=False,
             stop_gradient=False):
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
    print('\nBuilding model : Quantized ResNet50')
    pbar=tqdm(total=21)
    
    if BN_nbits is None:
        BN_nbits=nbits

    if BN_fbits is None:
        BN_fbits=fbits
        
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)            
            
    layer_BN_quantizer=build_layer_quantizer(BN_nbits,BN_fbits,rounding_method,overflow_mode,stop_gradient)
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(177)]
    else:
        pbar.set_postfix_str('Inject input fault')
    pbar.update()
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(177)]
    else:
        pbar.set_postfix_str('Inject output fault')
    pbar.update()
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(177)]
    else:
        pbar.set_postfix_str('Inject weight fault')
    pbar.update()

    
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
        
    pbar.set_postfix_str('building stage 1')

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = QuantizedConv2D(64, 
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='valid',
                        quantizers=layer_quantizer,
                        name='conv1',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[2],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[2],
                        weight_sa_fault_injection=weight_fault_dict_list[2],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=bn_axis, 
                                    name='bn_conv1',
                                    ifmap_sa_fault_injection=ifmap_fault_dict_list[3],
                                    ofmap_sa_fault_injection=ofmap_fault_dict_list[3],
                                    weight_sa_fault_injection=weight_fault_dict_list[3],
                                    quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    pbar.update()

    pbar.set_postfix_str('building stage 2 block a')
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), 
                   layer_quantizer=layer_quantizer, 
                   layer_BN_quantizer=layer_BN_quantizer, 
                   ifmap_fault_dict_list=ifmap_fault_dict_list[6:18],
                   ofmap_fault_dict_list=ofmap_fault_dict_list[6:18],
                   weight_fault_dict_list=weight_fault_dict_list[6:18],
                   mac_unit=mac_unit,
                   quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 2 block b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[18:28],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[18:28],
                       weight_fault_dict_list=weight_fault_dict_list[18:28],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 2 block c')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[28:38],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[28:38],
                       weight_fault_dict_list=weight_fault_dict_list[28:38],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 3 block a')
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', 
                   layer_quantizer=layer_quantizer, 
                   layer_BN_quantizer=layer_BN_quantizer, 
                   ifmap_fault_dict_list=ifmap_fault_dict_list[38:50],
                   ofmap_fault_dict_list=ofmap_fault_dict_list[38:50],
                   weight_fault_dict_list=weight_fault_dict_list[38:50],
                   mac_unit=mac_unit,
                   quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[50:60],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[50:60],
                       weight_fault_dict_list=weight_fault_dict_list[50:60],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[60:70],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[60:70],
                       weight_fault_dict_list=weight_fault_dict_list[60:70],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block d')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[70:80],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[70:80],
                       weight_fault_dict_list=weight_fault_dict_list[70:80],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 4 block a')    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', 
                   layer_quantizer=layer_quantizer, 
                   layer_BN_quantizer=layer_BN_quantizer, 
                   ifmap_fault_dict_list=ifmap_fault_dict_list[80:92],
                   ofmap_fault_dict_list=ofmap_fault_dict_list[80:92],
                   weight_fault_dict_list=weight_fault_dict_list[80:92],
                   mac_unit=mac_unit,
                   quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 4 block b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[92:102],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[92:102],
                       weight_fault_dict_list=weight_fault_dict_list[92:102],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 4 block c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[102:112],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[102:112],
                       weight_fault_dict_list=weight_fault_dict_list[102:112],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 4 block d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[112:122],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[112:122],
                       weight_fault_dict_list=weight_fault_dict_list[112:122],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 4 block e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[122:132],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[122:132],
                       weight_fault_dict_list=weight_fault_dict_list[122:132],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 4 block f')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[132:142],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[132:142],
                       weight_fault_dict_list=weight_fault_dict_list[132:142],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 5 block a')
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', 
                   layer_quantizer=layer_quantizer, 
                   layer_BN_quantizer=layer_BN_quantizer, 
                   ifmap_fault_dict_list=ifmap_fault_dict_list[142:154],
                   ofmap_fault_dict_list=ofmap_fault_dict_list[142:154],
                   weight_fault_dict_list=weight_fault_dict_list[142:154],
                   mac_unit=mac_unit,
                   quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 5 block b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[154:164],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[154:164],
                       weight_fault_dict_list=weight_fault_dict_list[154:164],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 5 block c')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', 
                       layer_quantizer=layer_quantizer, 
                       layer_BN_quantizer=layer_BN_quantizer, 
                       ifmap_fault_dict_list=ifmap_fault_dict_list[164:174],
                       ofmap_fault_dict_list=ofmap_fault_dict_list[164:174],
                       weight_fault_dict_list=weight_fault_dict_list[164:174],
                       mac_unit=mac_unit,
                       quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building output block')
    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = QuantizedFlatten()(x)
        x = QuantizedDense(classes, activation='softmax', name='fc1000',
                           quantizers=layer_quantizer,
                           ifmap_sa_fault_injection=ifmap_fault_dict_list[176],
                           ofmap_sa_fault_injection=ofmap_fault_dict_list[176],
                           weight_sa_fault_injection=weight_fault_dict_list[176],
                           mac_unit=mac_unit,
                           quant_mode=quant_mode,
                           last_layer=True)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')
    pbar.update()

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = backend.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='quantized_resnet50')
    
    pbar.set_postfix_str('Model Built')
    pbar.close()

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

def identity_block_fused_BN(input_tensor, 
                            kernel_size, 
                            filters, 
                            stage, 
                            block, 
                            layer_quantizer, 
                            quant_mode='hybrid',
                            ifmap_fault_dict_list=None, 
                            ofmap_fault_dict_list=None, 
                            weight_fault_dict_list=None
                            mac_unit=None):
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
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(7)]
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(7)]
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(7)]
    

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1),
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2a',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[0],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[0],
                        weight_sa_fault_injection=weight_fault_dict_list[0],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(input_tensor)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size,
                        quantizers=layer_quantizer,
                        padding='same', 
                        name=conv_name_base + '2b',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[2],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[2],
                        weight_sa_fault_injection=weight_fault_dict_list[2],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2c',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[4],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[4],
                        weight_sa_fault_injection=weight_fault_dict_list[4],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_fused_BN(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               layer_quantizer,
               strides=(2, 2),
               quant_mode='hybrid',
               ifmap_fault_dict_list=None, 
               ofmap_fault_dict_list=None, 
               weight_fault_dict_list=None,
               mac_unit=None):
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
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(8)]
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(8)]
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(8)]
    

    x = QuantizedConv2D(filters1, 
                        kernel_size=(1, 1), 
                        strides=strides,
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2a',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[0],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[0],
                        weight_sa_fault_injection=weight_fault_dict_list[0],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(input_tensor)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters2, 
                        kernel_size=kernel_size, 
                        padding='same',
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2b',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[2],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[2],
                        weight_sa_fault_injection=weight_fault_dict_list[2],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)

    x = QuantizedConv2D(filters3, 
                        kernel_size=(1, 1), 
                        quantizers=layer_quantizer,
                        name=conv_name_base + '2c',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[4],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[4],
                        weight_sa_fault_injection=weight_fault_dict_list[4],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)

    shortcut = QuantizedConv2D(filters3, 
                               kernel_size=(1, 1), 
                               strides=strides,
                               quantizers=layer_quantizer,
                               name=conv_name_base + '1',
                               ifmap_sa_fault_injection=ifmap_fault_dict_list[5],
                               ofmap_sa_fault_injection=ofmap_fault_dict_list[5],
                               weight_sa_fault_injection=weight_fault_dict_list[5],
                               mac_unit=mac_unit,
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
             quant_mode='hybrid',
             ifmap_fault_dict_list=None, 
             ofmap_fault_dict_list=None, 
             weight_fault_dict_list=None,
             mac_unit=None,
             overflow_mode=False,
             stop_gradient=False,):
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
    print('\nBuilding model : Quantized ResNet50')
    pbar=tqdm(total=21)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for _ in range(124)]
    else:
        pbar.set_postfix_str('Inject input fault')
    pbar.update()
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for _ in range(124)]
    else:
        pbar.set_postfix_str('Inject output fault')
    pbar.update()
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for _ in range(124)]
    else:
        pbar.set_postfix_str('Inject weight fault')
    pbar.update()

    
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
        
    pbar.set_postfix_str('building stage 1')

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = QuantizedConv2D(64, 
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='valid',
                        quantizers=layer_quantizer,
                        name='conv1',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[2],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[2],
                        weight_sa_fault_injection=weight_fault_dict_list[2],
                        mac_unit=mac_unit,
                        quant_mode=quant_mode)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    pbar.update()

    pbar.set_postfix_str('building stage 2 block a')
    x = conv_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), 
                            layer_quantizer=layer_quantizer, 
                            ifmap_fault_dict_list=ifmap_fault_dict_list[5:13],
                            ofmap_fault_dict_list=ofmap_fault_dict_list[5:13],
                            weight_fault_dict_list=weight_fault_dict_list[5:13],
                            mac_unit=mac_unit,
                            quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 2 block b')
    x = identity_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='b', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[13:20],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[13:20],
                                weight_fault_dict_list=weight_fault_dict_list[13:20],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 2 block c')
    x = identity_block_fused_BN(x, 3, [64, 64, 256], stage=2, block='c', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[20:27],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[20:27],
                                weight_fault_dict_list=weight_fault_dict_list[20:27],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 3 block a')
    x = conv_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='a', 
                            layer_quantizer=layer_quantizer, 
                            ifmap_fault_dict_list=ifmap_fault_dict_list[27:35],
                            ofmap_fault_dict_list=ofmap_fault_dict_list[27:35],
                            weight_fault_dict_list=weight_fault_dict_list[27:35],
                            mac_unit=mac_unit,
                            quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block b')
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='b', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[35:42],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[35:42],
                                weight_fault_dict_list=weight_fault_dict_list[35:42],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block c')
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='c', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[42:49],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[42:49],
                                weight_fault_dict_list=weight_fault_dict_list[42:49],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block d')
    x = identity_block_fused_BN(x, 3, [128, 128, 512], stage=3, block='d', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[49:56],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[49:56],
                                weight_fault_dict_list=weight_fault_dict_list[49:56],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 4 block a')    
    x = conv_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='a', 
                            ifmap_fault_dict_list=ifmap_fault_dict_list[56:64],
                            ofmap_fault_dict_list=ofmap_fault_dict_list[56:64],
                            weight_fault_dict_list=weight_fault_dict_list[56:64],
                            mac_unit=mac_unit,
                            layer_quantizer=layer_quantizer, 
                            quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block b')
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='b', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[64:71],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[64:71],
                                weight_fault_dict_list=weight_fault_dict_list[64:71],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block c')
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='c', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[71:78],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[71:78],
                                weight_fault_dict_list=weight_fault_dict_list[71:78],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block d')
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='d', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[78:85],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[78:85],
                                weight_fault_dict_list=weight_fault_dict_list[78:85],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block e')
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='e', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[85:92],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[85:92],
                                weight_fault_dict_list=weight_fault_dict_list[85:92],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 3 block f')
    x = identity_block_fused_BN(x, 3, [256, 256, 1024], stage=4, block='f', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[92:99],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[92:99],
                                weight_fault_dict_list=weight_fault_dict_list[92:99],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building stage 5 block a')
    x = conv_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='a', 
                            ifmap_fault_dict_list=ifmap_fault_dict_list[99:107],
                            ofmap_fault_dict_list=ofmap_fault_dict_list[99:107],
                            weight_fault_dict_list=weight_fault_dict_list[99:107],
                            mac_unit=mac_unit,
                            layer_quantizer=layer_quantizer, 
                            quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 5 block b')
    x = identity_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='b', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[107:114],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[107:114],
                                weight_fault_dict_list=weight_fault_dict_list[107:114],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()
    pbar.set_postfix_str('building stage 5 block c')
    x = identity_block_fused_BN(x, 3, [512, 512, 2048], stage=5, block='c', 
                                layer_quantizer=layer_quantizer, 
                                ifmap_fault_dict_list=ifmap_fault_dict_list[114:121],
                                ofmap_fault_dict_list=ofmap_fault_dict_list[114:121],
                                weight_fault_dict_list=weight_fault_dict_list[114:121],
                                mac_unit=mac_unit,
                                quant_mode=quant_mode)
    pbar.update()

    pbar.set_postfix_str('building output block')
    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = QuantizedFlatten()(x)
        x = QuantizedDense(classes, activation='softmax', name='fc1000',
                           quantizers=layer_quantizer,
                           ifmap_sa_fault_injection=ifmap_fault_dict_list[123],
                           ofmap_sa_fault_injection=ofmap_fault_dict_list[123],
                           weight_sa_fault_injection=weight_fault_dict_list[123],
                           mac_unit=mac_unit,
                           quant_mode=quant_mode,
                           last_layer=True)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')
    pbar.update()


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = backend.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='quantized_resnet50_fusedBN')
    
    pbar.set_postfix_str('Model Built')
    pbar.close()

    # load weights
    if weights is not None:
        model.load_weights(weights)

    return model


