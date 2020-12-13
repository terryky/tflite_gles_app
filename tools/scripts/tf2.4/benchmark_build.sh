#!/bin/sh

# -----------------------------------------------
#  Build Script for TFLite Model Benchmark Tool
#
#  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
# -----------------------------------------------

# [install jdk]
#   $ sudo apt install default-jdk
#
# [install bazel on aarch64 target (Raspi/Jetson)]
#   $ git clone https://github.com/PINTO0309/Bazel_bin
#   $ cd Bazel_bin/3.1.0/Raspbian_Debian_Buster_aarch64/openjdk-8-jdk
#   $ ./install.sh
#
# [install bazel on x64_64]
#   $ wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
#   $ chmod 755 bazel-3.1.0-installer-linux-x86_64.sh
#   $ sudo ./bazel-3.1.0-installer-linux-x86_64.sh

export TENSORFLOW_DIR=${HOME}/work/tensorflow

cd ${TENSORFLOW_DIR}

# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model

