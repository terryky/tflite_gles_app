#!/bin/sh

### at first, you need to clone Tensorflow source code.
#
# cd ~/work/
# https://github.com/tensorflow/tensorflow.git
# cd tensorflow
# git checkout r2.0
# 

make clean

make TARGET_ENV=raspi4 TFLITE_DELEGATE=GL_DELEGATE
