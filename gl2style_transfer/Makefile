include ../Makefile.env

TARGET = gl2style_transfer

SRCS = 
SRCS += main.c
SRCS += tflite_style_transfer.cpp
SRCS += ../common/assertgl.c
SRCS += ../common/assertegl.c
SRCS += ../common/util_egl.c
SRCS += ../common/util_shader.c
SRCS += ../common/util_matrix.c
SRCS += ../common/util_texture.c
SRCS += ../common/util_render2d.c
SRCS += ../common/util_debugstr.c
SRCS += ../common/util_pmeter.c
SRCS += ../common/winsys/$(WINSYS_SRC).c

OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

INCLUDES += -I../common/
INCLUDES += -I../common/winsys/
INCLUDES += -I../third_party/

LDFLAGS  +=
LIBS     += -pthread

# for V4L2 camera capture
CFLAGS   += -DUSE_INPUT_CAMERA_CAPTURE
SRCS     += camera_capture.c
SRCS     += ../common/util_v4l2.c
SRCS     += ../common/util_drm.c
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


# ---------------------
#  for TFLite GPU GL Delegate
# ---------------------

#LDFLAGS  += -L$(TENSORFLOW_DIR)/bazel-bin/tensorflow/lite/delegates/gpu
#LIBS     += -ltensorflowlite_gpu_gl
#LIBS     += -lgl_delegate



# ---------------------
#  for TFLite GPU GL Delegate with SSBO binding.
# ---------------------
#CFLAGS   += -DUSE_GLES_31
#CFLAGS   += -DUSE_INPUT_SSBO
#SRCS     += ssbo_tensor.c


# ---------------------
#  for Quantized tflite model
# ---------------------
#CFLAGS   += -DUSE_QUANT_TFLITE_MODEL


include ../Makefile.include
