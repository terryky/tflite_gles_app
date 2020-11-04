#!/bin/sh
set -e
set -x

#
# This tflite is converted from the pretrained model of https://github.com/yu4u/age-gender-estimation.
# Please note the license of the dataset ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
#
#wget https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5

python3 -m tf2onnx.convert \
    --keras EfficientNetB3_224_weights.11-3.44.hdf5 \
    --output EfficientNetB3_224_weights.11-3.44.onnx \
    --inputs input_1:0[1,224,224,3]
