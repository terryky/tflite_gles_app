#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.2
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}
export SCRIPT_DIR=`dirname $(realpath $0)`

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# apply patch for GPU Delegate
PATCH_FILE=${SCRIPT_DIR}/tensorflow_tf22_enable_gpu_delegate.diff
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
bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so 2>&1 | tee -a log_build_delegate.txt


# reuse bazel products for make.
cd ${TENSORFLOW_DIR}
ln -s ./bazel-bin/../../../external .
cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h tensorflow/lite/delegates/gpu/cl/

cp -rf ./external/flatbuffers/* tensorflow/lite/tools/make/downloads/flatbuffers/
cp -rf ./external/com_google_absl/* tensorflow/lite/tools/make/downloads/absl/

# build TensorFlow Lite library (libtensorflow-lite.a)
./tensorflow/lite/tools/make/build_rpi_lib.sh | tee -a log_build_libtflite_gpu_delegate.txt

echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l tensorflow/lite/tools/make/gen/rpi_armv7l/lib/
#ls -l bazel-bin/tensorflow/lite/delegates/gpu/



