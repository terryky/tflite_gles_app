#/bin/sh
set -e
set -x

python3 -m tf2onnx.convert	\
	--graphdef model-mobilenet_v1_101_257.pb \
	--output model-mobilenet_v1_101_257.onnx \
	--inputs image:0 \
	--inputs-as-nchw image:0 \
	--outputs heatmap:0,offset_2:0,displacement_fwd_2:0,displacement_bwd_2:0 \
	--fold_const	\
	--opset 10

