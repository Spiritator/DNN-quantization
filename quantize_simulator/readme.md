# Quantize Simulator

Use the digital circuit algorithm simulation method to simulate qunatized DNN by keras custom layer.

## now support:
### quantized layer
-   Conv2D
-   Dense
-   BatchNormalization
-   Depthwise Conv2D layer

### dataset
-   Mnist
-   CiFar10
- ImageDataGenerator
- ISLVRC2012 validation set in ImageDataGenerator format available [https://drive.google.com/open?id=1FW1N4AfYS8dKdqCYCo29dJl_W6fb2r2b]

### quantization
- you can specify weight word length and factorial bits
- nearest rounding
- zero rounding
- down rounding
- stochastic rounding

## future work
- support inject stuck at fault for the bit in arbitary neural network parameter
= support fault model to generate coressponding fault output bit of MAC calculation

## reference:


BertMoon https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow

TensorQuant by Dominik Marek Loroch, Norbert When, Franz-Josef Pfreundt, Janis Keuper.
https://arxiv.org/abs/1710.05758
