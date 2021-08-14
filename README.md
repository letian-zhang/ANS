## Autodidactic Neurosurgeon Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning
Autodidactic Neurosurgeon (ANS) is an online learning module to automatically learn the optimal DNN partition point on the fly. The details of ANS are in our WWW21 paper "Autodidactic Neurosurgeon Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning".

### PyTorch
We modify the *forward* function in the PyTorch to partition the DNN model.

You can run "**vgg16.py**" as an example to see the partition.

### TensorFlow 2.0+
Although we don't provide the code for TensorFlow, however, you can modify the *\__call\__* function in the TensorFlow to partition the DNN model.

### Two examples:
- vgg16
- tiny yolo v2
  - Download tiny yolo weight, here https://pjreddie.com/media/files/yolov2-tiny-voc.weights
  - Put yolov2-tiny-voc.weights in the folder "models"

### How to run code:
- First run *edge_server_main.py* on edge server
- Then run *client_camera_main.py* on Nvidia Jetson TX2
