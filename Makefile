#
# > make TARGET_ENV=x11
# > make TARGET_ENV=raspi
#
all:
	make -C gl2blazeface
	make -C gl2detection
	make -C gl2facemesh
	make -C gl2handpose
	make -C gl2posenet
	make -C gl2segmentation
	make -C gl2style_transfer

clean:
	make -C gl2blazeface clean
	make -C gl2detection clean
	make -C gl2facemesh clean
	make -C gl2handpose clean
	make -C gl2posenet clean
	make -C gl2segmentation clean
	make -C gl2style_transfer clean
