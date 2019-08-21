#
# > make TARGET_ENV=x11
# > make TARGET_ENV=raspi
#
all:
	make -C list_egl_configs
	make -C gl2tri
	make -C gl2teapot
	make -C gl2detection
	make -C gl2posenet
	make -C gl2segmentation

clean:
	make -C list_egl_configs clean
	make -C gl2tri clean
	make -C gl2teapot clean
	make -C gl2detection clean
	make -C gl2posenet clean
	make -C gl2segmentation clean
