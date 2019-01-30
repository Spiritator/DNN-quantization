# DNN Fault Simulator

DNN quantize conversion and accuracy loss evaluation

## Quantize Simulator

Use the digital circuit algorithm simulation method to simulate qunatized DNN by keras custom layer.

## Fault Simulator

Use bitwise operation on multidimension array or Tensor to create fault injection on DNN.

### now support:
#### quantized layer
-   Conv2D
-   Dense
-   BatchNormalization
-   Depthwise Conv2D layer

#### dataset
-   Mnist
-   CiFar10
- ImageDataGenerator
- ISLVRC2012 validation set in ImageDataGenerator format available  [Download Link](https://drive.google.com/open?id=1FW1N4AfYS8dKdqCYCo29dJl_W6fb2r2b)

#### quantization
- you can specify weight word length and factorial bits
- nearest rounding
- zero rounding
- down rounding
- stochastic rounding

#### fault simulation
- interconnection fault (SA fault)
- memory fault (SA fault)

#### fault list generation
- uniform distributed fault
- poisson distributed fault

#### approximation
- number of computaion estimation
- number of bits in computaion

### future work
- support hardware accelerator logic fault simulation
- support memory coupling fault simulation
- support pooling fault simulation
- sppport different fault distribution
= support fault model to generate coressponding fault output bit of MAC calculation

### reference:


BertMoon https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow

TensorQuant by Dominik Marek Loroch, Norbert When, Franz-Josef Pfreundt, Janis Keuper.
https://arxiv.org/abs/1710.05758

## Other files
Trivial conversions and qunatization codes include python and matlab
