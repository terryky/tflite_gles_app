#/bin/sh
set -e
set -x

python3 -m tf2onnx.convert \
    --saved-model saved_model_keras_480x640 \
    --output dbface_keras_480x640_float32_nhwc.onnx

python3 -m tf2onnx.convert \
    --saved-model saved_model_keras_256x256 \
    --output dbface_keras_256x256_float32_nhwc.onnx
