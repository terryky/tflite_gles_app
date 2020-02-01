#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.0
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}
export SCRIPT_DIR=`dirname $(realpath $0)`

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# apply patch for GPU Delegate
PATCH_FILE=${SCRIPT_DIR}/tensorflow_tf20_enable_gpu_delegate.diff
patch -p1 < ${PATCH_FILE}


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


# build GPU Delegate library (libtensorflowlite_gpu_gl.so)
bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so 2>&1 | tee -a log_build_delegate.txt


# reuse bazel products for make.
cd ${TENSORFLOW_DIR}
ln -s ./bazel-bin/../../../external .
#cp bazel-out/k8-opt/genfiles/tensorflow/lite/delegates/gpu/gl/metadata_generated.h       ./tensorflow/lite/delegates/gpu/gl/
#cp bazel-out/k8-opt/genfiles/tensorflow/lite/delegates/gpu/gl/common_generated.h         ./tensorflow/lite/delegates/gpu/gl/
#cp bazel-out/k8-opt/genfiles/tensorflow/lite/delegates/gpu/gl/workgroups_generated.h     ./tensorflow/lite/delegates/gpu/gl/
#cp bazel-out/k8-opt/genfiles/tensorflow/lite/delegates/gpu/gl/compiled_model_generated.h ./tensorflow/lite/delegates/gpu/gl/

cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/gl/metadata_generated.h       ./tensorflow/lite/delegates/gpu/gl/
cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/gl/common_generated.h         ./tensorflow/lite/delegates/gpu/gl/
cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/gl/workgroups_generated.h     ./tensorflow/lite/delegates/gpu/gl/
cp bazel-out/k8-opt/bin/tensorflow/lite/delegates/gpu/gl/compiled_model_generated.h ./tensorflow/lite/delegates/gpu/gl/


# build TensorFlow Lite library (libtensorflow-lite.a)
./tensorflow/lite/tools/make/build_rpi_lib.sh | tee -a log_build_libtflite_gpu_delegate.txt


echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
ls -l tensorflow/lite/tools/make/gen/rpi_armv7l/lib/
#ls -l bazel-bin/tensorflow/lite/delegates/gpu/



