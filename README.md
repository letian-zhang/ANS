## Autodidactic Neurosurgeon Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning
#### Autodidactic Neurosurgeon (ANS) is an online learning module to automaticallly learn the optimal DNN partition point on-the-fly. The detials of ANS is in our WWW21 paper "Autodidactic Neurosurgeon Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning".

### PyTorch
We modify the *forward* function in the PyTorch DNN model to partition the inference.

You can run the "**vgg16.py**" as an example to see the partition.

### TensorFlow 2.0+
Although we don't provide the code for TensorFlow, however you can modify the *\__call\__* fuction in the TensorFlow DNN model to partition the inference.
