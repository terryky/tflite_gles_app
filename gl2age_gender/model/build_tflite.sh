#!/bin/sh

#
# This tflite is converted from the pretrained model of https://github.com/yu4u/age-gender-estimation.
# Please note the license of the dataset ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
#
wget https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5

tflite_convert \
    --keras_model_file=EfficientNetB3_224_weights.11-3.44.hdf5 \
    --output_file=EfficientNetB3_224_weights.11-3.44.tflite
  