MAKETOP = $(realpath ..)
include $(MAKETOP)/Makefile.env

TARGET = gl2handpose

SRCS = 
SRCS += main.c
SRCS += render_handpose.c
SRCS += tflite_handpose.cpp
SRCS += shapes.c
SRCS += custom_ops/transpose_conv_bias.cc
SRCS += $(MAKETOP)/common/assertgl.c
SRCS += $(MAKETOP)/common/assertegl.c
SRCS += $(MAKETOP)/common/util_egl.c
SRCS += $(MAKETOP)/common/util_shader.c
SRCS += $(MAKETOP)/common/util_matrix.c
SRCS += $(MAKETOP)/common/util_texture.c
SRCS += $(MAKETOP)/common/util_render2d.c
SRCS += $(MAKETOP)/common/util_debugstr.c
SRCS += $(MAKETOP)/common/util_pmeter.c
SRCS += $(MAKETOP)/common/util_tflite.cpp
SRCS += $(MAKETOP)/common/winsys/$(WINSYS_SRC).c

OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

LDFLAGS  +=
LIBS     += -pthread

# for V4L2 camera capture
CFLAGS   += -DUSE_INPUT_CAMERA_CAPTURE
SRCS     += camera_capture.c
SRCS     += $(MAKETOP)/common/util_v4l2.c
SRCS     += $(MAKETOP)/common/util_drm.c
LIBS     += -ldrm


# ---------------------
#  for TFLite
# ---------------------
TENSORFLOW_DIR = $(HOME)/work/tensorflow

INCLUDES += -I$(TENSORFLOW_DIR)
INCLUDES += -I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include
INCLUDES += -I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/absl
INCLUDES += -I$(TENSORFLOW_DIR)/external/flatbuffers/include
INCLUDES += -I$(TENSORFLOW_DIR)/external/com_google_absl
LDFLAGS  += -L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/linux_x86_64/lib/
LDFLAGS  += -L$(TENSORFLOW_DIR)/
LDFLAGS  += -L./
LIBS     += -ltensorflow-lite -ldl

LDFLAGS  += -Wl,--allow-multiple-definition

include ../Makefile.include
