#
# > make TARGET_ENV=x11
# > make TARGET_ENV=raspi
#
all:
	$(MAKE) -C gl2blazeface
	$(MAKE) -C gl2classification
	$(MAKE) -C gl2detection
	$(MAKE) -C gl2facemesh
	$(MAKE) -C gl2hair_segmentation
	$(MAKE) -C gl2handpose
	$(MAKE) -C gl2objectron
	$(MAKE) -C gl2posenet
	$(MAKE) -C gl2segmentation
	$(MAKE) -C gl2style_transfer

clean:
	$(MAKE) -C gl2blazeface clean
	$(MAKE) -C gl2classification clean
	$(MAKE) -C gl2detection clean
	$(MAKE) -C gl2facemesh clean
	$(MAKE) -C gl2hair_segmentation clean
	$(MAKE) -C gl2handpose clean
	$(MAKE) -C gl2objectron clean
	$(MAKE) -C gl2posenet clean
	$(MAKE) -C gl2segmentation clean
	$(MAKE) -C gl2style_transfer clean
