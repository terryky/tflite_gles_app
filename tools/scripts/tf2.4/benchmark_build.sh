#!/bin/sh

# -----------------------------------------------
#  Build Script for TFLite Model Benchmark Tool
#
#  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
# -----------------------------------------------

export TENSORFLOW_DIR=~/work/tensorflow

cd ${TENSORFLOW_DIR}

# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model

