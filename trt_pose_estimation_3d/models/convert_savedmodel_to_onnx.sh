#/bin/sh
set -e
set -x

python3 -m tf2onnx.convert \
    --saved-model saved_model \
    --output human-pose-estimation-3d.onnx\
    --inputs data:0[1,256,448,3]


    