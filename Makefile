all:
	make -C list_egl_configs
	make -C gl2tri

clean:
	make -C list_egl_configs clean
	make -C gl2tri clean
