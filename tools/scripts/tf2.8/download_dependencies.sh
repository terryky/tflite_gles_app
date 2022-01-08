#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.8
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}


# install CMake 3.16 or higher
#sudo apt purge cmake
#wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
#wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.sh
#chmod 755 cmake-3.22.1-linux-x86_64.sh 
#sudo ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local


# -----------------------------------------------------------------------
#  Launch CMake to download external header files (flatbuffers, ...)
# -----------------------------------------------------------------------

# download all the build dependencies.
cd ${TENSORFLOW_DIR}
mkdir external
cd external
cmake ../tensorflow/lite -DCMAKE_FIND_DEBUG_MODE=1 2>&1 | tee -a log_cmake.txt

