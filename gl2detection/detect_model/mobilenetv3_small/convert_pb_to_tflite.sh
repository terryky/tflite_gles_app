#!/bin/bash
set -e
set -x

PB_FILE="frozen_inference_graph.pb"
INNODE_NAME="normalized_input_image_tensor"
OUTNODE_NAME=raw_outputs/box_encodings,raw_outputs/class_predictions
INNODE_SHAPE=1,320,320,3

TFLITE_FILE="ssd_mobilenet_v3_small_coco_float.tflite"

tflite_convert \
  --enable_v1_converter \
  --graph_def_file=${PB_FILE} \
  --output_file=${TFLITE_FILE} \
  --output_format=TFLITE \
  --input_arrays=${INNODE_NAME} \
  --output_arrays=${OUTNODE_NAME} \
  --input_shapes=${INNODE_SHAPE}


set +x
echo ""
echo "----------------------------------"
echo "[SUCCESS] " ${TFLITE_FILE}
set -x



