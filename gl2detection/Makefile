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
SRCS += ../common/util_render2d.c
SRCS += ../common/util_debugstr.c
SRCS += ../common/util_pmeter.c
SRCS += ../common/util_tflite.cpp
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

#
# for FFmpeg (libav) video decode
#
ENABLE_VDEC ?= false
#ENABLE_VDEC := true
ifeq ($(ENABLE_VDEC), true)
CFLAGS   += -DUSE_INPUT_VIDEO_DECODE
FFMPEG_LIBS=    libavdevice                        \
                libavformat                        \
                libavfilter                        \
                libavcodec                         \
                libswresample                      \
                libswscale                         \
                libavutil                          \

CFLAGS += $(shell pkg-config --cflags $(FFMPEG_LIBS))
LIBS   += $(shell pkg-config --libs   $(FFMPEG_LIBS)) -lm
SRCS   += video_decode.c
endif


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
