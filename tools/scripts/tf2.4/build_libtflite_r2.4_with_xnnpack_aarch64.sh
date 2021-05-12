#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.4
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone -b ${TENSORFLOW_VER} https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}


# install Bazel 3.1.0
#wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
#chmod 755 bazel-3.1.0-installer-linux-x86_64.sh
#sudo ./bazel-3.1.0-installer-linux-x86_64.sh


# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


# ---------------
#  Bazel build
# ---------------
# build with Bazel (libtensorflowlite.so)
bazel build -s -c opt --config=elinux_aarch64 --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.so  2>&1 | tee -a log_build_libtflite_bazel.txt

# Sparse Inference (experimental)
#bazel build -s -c opt --config=elinux_aarch64 --define tflite_with_xnnpack=true --define xnn_enable_sparse=true //tensorflow/lite:libtensorflowlite.so

# build GPU Delegate library (libdelegate.so)
bazel build -s -c opt --config=elinux_aarch64 --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so 2>&1 | tee -a log_build_delegate.txt


echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l bazel-bin/tensorflow/lite/
ls -l bazel-bin/tensorflow/lite/delegates/gpu/



