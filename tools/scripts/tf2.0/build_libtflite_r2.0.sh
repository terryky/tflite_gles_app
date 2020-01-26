#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.0
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# install Bazel 0.26.1
#wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
#chmod 755 bazel-0.26.1-installer-linux-x86_64.sh
#sudo ./bazel-0.26.1-installer-linux-x86_64.sh

# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


# download all the build dependencies.
./tensorflow/lite/tools/make/download_dependencies.sh 2>&1 | tee -a log_download_dependencies.txt


# apply a patch regarding "undefined reference to `rdft2d'"
#
#  ./tensorflow/lite/tools/make/Makefile
#    $(wildcard tensorflow/lite/tools/make/downloads/fft2d/fftsg.c) \
#  + $(wildcard tensorflow/lite/tools/make/downloads/fft2d/fftsg2d.c) \
#    $(wildcard tensorflow/lite/tools/make/downloads/flatbuffers/src/util.cpp)
sed -e '/fftsg.c/a tensorflow\/lite\/tools\/make\/downloads\/fft2d\/fftsg2d.c \\' -i ./tensorflow/lite/tools/make/Makefile

# build TensorFlow Lite library (libtensorflow-lite.a)
make -j 4  -f ./tensorflow/lite/tools/make/Makefile BUILD_WITH_NNAPI=false EXTRA_CXXFLAGS="-march=native" 2>&1 | tee -a log_build_libtflite.txt

# build GPU Delegate library (libtensorflowlite_gpu_gl.so)
bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so 2>&1 | tee -a log_build_delegate.txt



echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l tensorflow/lite/tools/make/gen/linux_x86_64/lib/
ls -l bazel-bin/tensorflow/lite/delegates/gpu/



