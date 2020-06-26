#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.3
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


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
#  Makefile build
# ---------------

# download all the build dependencies.
./tensorflow/lite/tools/make/download_dependencies.sh 2>&1 | tee -a log_download_dependencies.txt

# build TensorFlow Lite library (libtensorflow-lite.a)
./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv7l 2>&1 | tee -a log_build_libtflite_make.txt


# ---------------
#  Bazel build
# ---------------
# build with Bazel (libtensorflowlite.so)
bazel build -s -c opt --config=elinux_armhf //tensorflow/lite:libtensorflowlite.so 2>&1 | tee -a log_build_libtflite_bazel.txt

# build GPU Delegate library (libdelegate.so)
bazel build -s -c opt --config=elinux_armhf --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so 2>&1 | tee -a log_build_delegate.txt


echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l tensorflow/lite/tools/make/gen/rpi_armv7l/lib/
ls -l bazel-bin/tensorflow/lite/
ls -l bazel-bin/tensorflow/lite/delegates/gpu/



