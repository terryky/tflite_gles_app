include ../Makefile.env

TARGET = gl2detection

SRCS = 
SRCS += main.c
SRCS += tflite_detect.cpp
SRCS += detect_postprocess.cpp
SRCS += ../common/assertgl.c
SRCS += ../common/assertegl.c
SRCS += ../common/util_egl.c
SRCS += ../common/util_shader.c
SRCS += ../common/util_matrix.c
SRCS += ../common/util_texture.c
SRCS += ../common/util_image_jpg.c
SRCS += ../common/util_render2d.c
SRCS += ../common/util_debugstr.c
SRCS += ../common/util_pmeter.c
SRCS += ../common/winsys/$(WINSYS_SRC).c

OBJS += $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS)))

INCLUDES += -I../common/
INCLUDES += -I../common/winsys/


LDFLAGS  +=

# for JEPG texture
CFLAGS   += -DUSE_JPEG_TEXTURE
LIBS     += -lpthread
LIBS     += -ljpeg

# ---------------------
#  for TFLite
# ---------------------
TENSORFLOW_DIR = $(HOME)/work/tensorflow

INCLUDES += -I$(TENSORFLOW_DIR)
INCLUDES += -I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include
LDFLAGS  += -L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/linux_x86_64/lib/
LIBS     += -ltensorflow-lite -ldl

LDFLAGS  += -Wl,--allow-multiple-definition

include ../Makefile.include
