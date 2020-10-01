#
# > make TARGET_ENV=x11
# > make TARGET_ENV=raspi
#
all:
	$(MAKE) -C gl2animegan2
	$(MAKE) -C gl2blazeface
	$(MAKE) -C gl2blazepose
	$(MAKE) -C gl2blazepose_fullbody
	$(MAKE) -C gl2classification
	$(MAKE) -C gl2dbface
	$(MAKE) -C gl2detection
	$(MAKE) -C gl2face_segmentation
	$(MAKE) -C gl2facemesh
	$(MAKE) -C gl2hair_segmentation
	$(MAKE) -C gl2handpose
	$(MAKE) -C gl2iris_landmark
	$(MAKE) -C gl2objectron
	$(MAKE) -C gl2pose_estimation_3d
	$(MAKE) -C gl2posenet
	$(MAKE) -C gl2segmentation
	$(MAKE) -C gl2selfie2anime
	$(MAKE) -C gl2style_transfer
	$(MAKE) -C gl2text_detection

clean:
	$(MAKE) -C gl2animegan2 clean
	$(MAKE) -C gl2blazeface clean
	$(MAKE) -C gl2blazepose clean
	$(MAKE) -C gl2blazepose_fullbody clean
	$(MAKE) -C gl2classification clean
	$(MAKE) -C gl2dbface clean
	$(MAKE) -C gl2detection clean
	$(MAKE) -C gl2face_segmentation clean
	$(MAKE) -C gl2facemesh clean
	$(MAKE) -C gl2hair_segmentation clean
	$(MAKE) -C gl2handpose clean
	$(MAKE) -C gl2iris_landmark clean
	$(MAKE) -C gl2objectron clean
	$(MAKE) -C gl2pose_estimation_3d clean
	$(MAKE) -C gl2posenet clean
	$(MAKE) -C gl2segmentation clean
	$(MAKE) -C gl2selfie2anime clean
	$(MAKE) -C gl2style_transfer clean
	$(MAKE) -C gl2text_detection clean
