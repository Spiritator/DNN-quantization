# Quantize Simulator

Use the digital circuit algorithm simulation method to simulate qunatized DNN by keras custom layer.

## now support:
### quantized layer
-   Conv2D
-   Dense
-   BatchNormalization

### dataset
-   Mnist
-   CiFar10
- ImageDataGenerator

### quantization
- you can specify weight word length and factorial bits
- nearest rounding
- zero rounding
- down rounding
- stochastic rounding

## future work
- support depthwise Conv2D layer
- support pointwise Conv2D layer

## reference:


BertMoon https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow

TensorQuant by Dominik Marek Loroch, Norbert When, Franz-Josef Pfreundt, Janis Keuper.
https://arxiv.org/abs/1710.05758
