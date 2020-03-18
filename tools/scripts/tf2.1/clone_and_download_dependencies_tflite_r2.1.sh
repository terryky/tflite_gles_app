#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.1
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}
export SCRIPT_DIR=`dirname $(realpath $0)`

git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# apply patch for GPU Delegate
#PATCH_FILE=${SCRIPT_DIR}/tensorflow_tf21_tflite_download_dependencies.diff
#patch -p1 < ${PATCH_FILE}

# download all the build dependencies.
./tensorflow/lite/tools/make/download_dependencies.sh

