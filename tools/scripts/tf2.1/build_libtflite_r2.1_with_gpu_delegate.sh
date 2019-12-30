#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.1
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# apply patch for GPU Delegate
export SCRIPT_DIR=`dirname $0`
PATCH_FILE=${SCRIPT_DIR}/tensorflow_tf21_enable_gpu_delegate.diff
patch -p1 < ${PATCH_FILE}


# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


# download all the build dependencies.
./tensorflow/lite/tools/make/download_dependencies.sh 2>&1 | tee -a log_download_dependencies.txt


# build GPU Delegate library (libdelegate.a)
bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:delegate 2>&1 | tee -a log_build_delegate.txt


# reuse bazel products for make.
cd ${TENSORFLOW_DIR}
ln -s ./bazel-bin/../../../external .
cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h tensorflow/lite/delegates/gpu/cl/


# build TensorFlow Lite library (libtensorflow-lite.a)
make -j 4  -f ./tensorflow/lite/tools/make/Makefile BUILD_WITH_NNAPI=false EXTRA_CXXFLAGS="-march=native" 2>&1 | tee -a log_build_libtflite_gpu_delegate.txt


echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l tensorflow/lite/tools/make/gen/linux_x86_64/lib/

