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


# install Bazel 4.2.1
#wget https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-installer-linux-x86_64.sh
#chmod 755 bazel-4.2.1-installer-linux-x86_64.sh
#sudo ./bazel-4.2.1-installer-linux-x86_64.sh



# ---------------
#  Launch CMake to download external header files (flatbuffers, ...)
# ---------------

# download all the build dependencies.
cd ${TENSORFLOW_DIR}
mkdir external
cd external
cmake ../tensorflow/lite -DCMAKE_FIND_DEBUG_MODE=1 2>&1 | tee -a log_cmake.txt



# clean up bazel cache, just in case.
cd ${TENSORFLOW_DIR}
bazel clean

echo "----------------------------------------------------"
echo " (configure) press ENTER-KEY several times.         "
echo "----------------------------------------------------"
./configure


# ---------------
#  Bazel build
# ---------------
# build with Bazel (libtensorflowlite.so)
bazel build -s -c opt --config=elinux_aarch64 //tensorflow/lite:libtensorflowlite.so 2>&1 | tee -a log_build_libtflite_bazel.txt

# build GPU Delegate library (libdelegate.so)
bazel build -s -c opt --config=elinux_aarch64 --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so 2>&1 | tee -a log_build_delegate.txt


echo "----------------------------------------------------"
echo " build success."
echo "----------------------------------------------------"

cd ${TENSORFLOW_DIR}
#ls -l tensorflow/lite/tools/make/gen/linux_aarch64/lib/
ls -l bazel-bin/tensorflow/lite/
ls -l bazel-bin/tensorflow/lite/delegates/gpu/


mkdir -p ~/lib/tf2.8_aarch64
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ~/lib/tf2.8_aarch64
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ~/lib/tf2.8_aarch64

