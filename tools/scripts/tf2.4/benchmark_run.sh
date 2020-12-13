#!/bin/sh
set -e 
#set -x

# -----------------------------------------------
#  Execute Script for TFLite Model Benchmark Tool
#
#  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
# -----------------------------------------------

export TENSORFLOW_DIR=${HOME}/work/tensorflow


if [ $# -ne 1 ]; then
  echo "usage: benchmark_run.sh model.tflite" 1>&2
  exit 1
fi


BENCH_ARGS="--graph=$1"
BENCH_ARGS=${BENCH_ARGS}" --warmup_runs=10"
BENCH_ARGS=${BENCH_ARGS}" --num_runs=100"
BENCH_ARGS=${BENCH_ARGS}" --enable_op_profiling=true"

# use multi threading
BENCH_ARGS=${BENCH_ARGS}" --num_threads=4"

# use GPU
#BENCH_ARGS=${BENCH_ARGS}" --use_gpu=true"
#BENCH_ARGS=${BENCH_ARGS}" --gpu_precision_loss_allowed=true"

# use XNNPACK 
#BENCH_ARGS=${BENCH_ARGS}" --use_xnnpack=true"



set -x
${TENSORFLOW_DIR}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model ${BENCH_ARGS}
