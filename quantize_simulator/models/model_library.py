'''
reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-Tensorflow

@author: Yung-Yu Tsai

'''

from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras.regularizers import l2
from keras import metrics
from keras import backend as K
import numpy as np

from layers.quantized_layers import QuantizedConv2D, QuantizedDense, QuantizedBatchNormalization
from layers.quantized_ops import quantized_relu as quantize_op


def quantized_lenet5(nbits=8, fbits=4, rounding_method='nearest', input_shape=(28,28,1), num_classes=10):
    
    print('Building model : Quantized Lenet 5')
    
    model = Sequential()
    model.add(QuantizedConv2D(filters=16,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(5,5),
                              padding='same',
                              strides=(1, 1),
                              input_shape=input_shape,
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(QuantizedConv2D(filters=36,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(5,5),
                              padding='same',
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(QuantizedDense(128,
                             H=1,
                             nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                             activation='relu'))
    model.add(QuantizedDense(num_classes,
                             H=1,
                             nb=nbits,
                             fb=fbits, 
                             rounding_method=rounding_method,
                             activation='softmax'))
#    model.summary()
#    model.compile(loss='categorical_crossentropy',
#    			  optimizer='adam',metrics=['accuracy'])

    model.summary()

    return model

def quantized_4C2F(nbits=8, fbits=4, rounding_method='nearest', input_shape=(32,32,3), num_classes=10):
    
    print('Building model : Quantized 4C2F CNN')
    
    model = Sequential()
    model.add(QuantizedConv2D(filters=32,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(3, 3),
                              padding='same',
                              strides=(1, 1),
                              input_shape=input_shape,
                              activation='relu'))
    model.add(QuantizedConv2D(filters=32,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(QuantizedConv2D(filters=64,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(3, 3),
                              padding='same',
                              strides=(1, 1),
                              activation='relu'))
    model.add(QuantizedConv2D(filters=64,
                              H=1,
                              nb=nbits,
                              fb=fbits, 
                              rounding_method=rounding_method,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(QuantizedDense(512,
                             H=1,
                             nb=nbits,
                             fb=fbits, 
                             rounding_method=rounding_method,
                             activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(QuantizedDense(num_classes,
                             H=1,
                             nb=nbits,
                             fb=fbits, 
                             rounding_method=rounding_method,
                             activation='softmax'))
    
#    model.compile(loss='categorical_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy', top2_acc])
    
    model.summary()

    return model

def quantized_droneNet(version, nbits=8, fbits=4, BN_nbits=10, BN_fbits=5, rounding_method='nearest', inputs=None,  include_top=True, classes=10, *args, **kwargs):
    if inputs is None :
        if K.image_data_format() == 'channels_first':
            input_shape = Input(shape=(3, 224, 224))
        else:
            input_shape = Input(shape=(224, 224, 3))
    else:
        input_shape=inputs
        
    print('Building model : Quantized DroneNet V%d at input shape'%version,end=' ')
    print(input_shape.shape)

    outputs = []

    x = QuantizedConv2D(filters=32,
                        H=1,
                        nb=nbits,
                        fb=fbits,
                        rounding_method=rounding_method,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False)(input_shape)
    x = QuantizedBatchNormalization(H=1,
                                    nb=BN_nbits,
                                    fb=BN_fbits,
                                    rounding_method=rounding_method)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    outputs.append(x)

    for i in range(3):
        x = QuantizedConv2D(filters=64*(2**i),
                            H=1,
                            nb=nbits,
                            fb=fbits,
                            rounding_method=rounding_method,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            use_bias=False)(x)
        x = QuantizedBatchNormalization(H=1,
                                        nb=BN_nbits,
                                        fb=BN_fbits,
                                        rounding_method=rounding_method)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        outputs.append(x)

    x = QuantizedConv2D(filters=256,
                        H=1,
                        nb=nbits,
                        fb=fbits,
                        rounding_method=rounding_method,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=False)(x)
    x = QuantizedBatchNormalization(H=1,
                           nb=BN_nbits,
                           fb=BN_fbits,
                           rounding_method=rounding_method)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    

    if include_top:
        x = Flatten()(x)
        if version == 1:
            x = QuantizedDense(1024,
                               H=1,
                               nb=nbits,
                               fb=fbits,
                               rounding_method=rounding_method,
                               activation='sigmoid')(x)
            x = QuantizedBatchNormalization(H=1,
                                            nb=BN_nbits,
                                            fb=BN_fbits,
                                            rounding_method=rounding_method)(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
        x = QuantizedDense(classes,
                           H=1,
                           nb=nbits,
                           fb=fbits,
                           rounding_method=rounding_method,
                           activation='sigmoid')(x)
        return Model(inputs=input_shape, outputs=x, *args, **kwargs)
    else:
        return Model(inputs=input_shape, outputs=outputs, *args, **kwargs)
    

#def convert_original_weight_layer_name(original_weight_name,quantized_weight_name=None):
#    
#    def load_attributes_from_hdf5_group(group, name):
#        """Loads attributes of the specified name from the HDF5 group.
#    
#        This method deals with an inherent problem
#        of HDF5 file which is not able to store
#        data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
#    
#        # Arguments
#            group: A pointer to a HDF5 group.
#            name: A name of the attributes to load.
#    
#        # Returns
#            data: Attributes data.
#        """
#        if name in group.attrs:
#            data = [n.decode('utf8') for n in group.attrs[name]]
#        else:
#            data = []
#            chunk_id = 0
#            while ('%s%d' % (name, chunk_id)) in group.attrs:
#                data.extend([n.decode('utf8')
#                             for n in group.attrs['%s%d' % (name, chunk_id)]])
#                chunk_id += 1
#        return data
#    
#    
#    o_weight_f = h5py.File(original_weight_name,'r')
#    if quantized_weight_name is None:
#        quantized_weight_name=original_weight_name[:-3]+'_quantized.h5'
#        if os.path.isfile(quantized_weight_name):
#            o_weight_f.close()
#            return quantized_weight_name
#        else:
#            q_weight_f = h5py.File(quantized_weight_name,'w')
#    else:
#        if os.path.isfile(quantized_weight_name):
#            o_weight_f.close()
#            return quantized_weight_name
#        else:
#            q_weight_f = h5py.File(quantized_weight_name,'w')
#        
#        
#    if 'keras_version' in o_weight_f.attrs:
#            original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
#    else:
#        original_keras_version = '1'
#    if 'backend' in o_weight_f.attrs:
#        original_backend = o_weight_f.attrs['backend'].decode('utf8')
#    else:
#        original_backend = None
#        
#    
#    layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
#    filtered_layer_names = []
#    for layer_name in layer_names:
#        o_group = o_weight_f[layer_name]
#        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
#        if weight_names:
#            filtered_layer_names.append(layer_name)
#            
#    quantized_layer_names = []
#    for layer in layer_names:
#        if layer in filtered_layer_names:
#            quantized_layer_names.append('quantized_'+layer)
#        else:
#            quantized_layer_names.append(layer)
#            
#    q_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in quantized_layer_names])
#    q_weight_f.attrs.create('backend',original_backend.encode('utf8'))
#    q_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))
#    
#    for layer_iter, layer_name in enumerate(layer_names):
#        o_group = o_weight_f[layer_name]
#        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
#        weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
#        quantized_layer = q_weight_f.create_group(quantized_layer_names[layer_iter])
#        if layer_name in filtered_layer_names:
#            quantized_layer.attrs.create('weight_names',[('quantized_'+temp).encode('utf8') for temp in weight_names])
#        else:
#            quantized_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
#        quantized_sublayer = quantized_layer.create_group(quantized_layer_names[layer_iter])
#        
#        for weight_iter, weight_name in enumerate(weight_names):
#            quantized_sublayer.create_dataset(weight_name[len(layer_name)+1:],weight_values[weight_iter].shape,weight_values[weight_iter].dtype,weight_values[weight_iter])
#        
#        
#    o_weight_f.close()
#    q_weight_f.close()
#    
#    return quantized_weight_name
