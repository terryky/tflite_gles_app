MAKETOP = $(realpath ..)
include $(MAKETOP)/Makefile.env

TARGET = gl2pose_estimation_3d

SRCS = 
SRCS += main.c
SRCS += tflite_pose3d.cpp
SRCS += render_pose3d.c
SRCS += shapes.c
SRCS += touch_event.c
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
CFLAGS   += -DUSE_INPUT_CAMERA_CAPTURE2
SRCS     += $(MAKETOP)/common/util_camera_capture.c
SRCS     += $(MAKETOP)/common/util_v4l2.c
SRCS     += $(MAKETOP)/common/util_drm.c
LIBS     += -ldrm

#
# for FFmpeg (libav) video decode
#
ifeq ($(ENABLE_VDEC), true)
CFLAGS   += -DUSE_INPUT_VIDEO_DECODE
CFLAGS   += -DUSE_INPUT_VIDEO_DECODE2
FFMPEG_LIBS=    libavdevice                        \
                libavformat                        \
                libavfilter                        \
                libavcodec                         \
                libswresample                      \
                libswscale                         \
                libavutil                          \

CFLAGS += $(shell pkg-config --cflags $(FFMPEG_LIBS))
LIBS   += $(shell pkg-config --libs   $(FFMPEG_LIBS)) -lm
SRCS   += $(MAKETOP)/common/util_video_decode.c
endif

# ---------------------
#  for ImGui
# ---------------------
ENABLE_IMGUI = true
ifeq ($(ENABLE_IMGUI), true)
CFLAGS   += -DUSE_IMGUI
CFLAGS   += -DIMGUI_IMPL_OPENGL_ES2

SRCS += render_imgui.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui_draw.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui_widgets.cpp
SRCS += $(MAKETOP)/third_party/imgui/examples/imgui_impl_opengl3.cpp

INCLUDES += -I$(MAKETOP)/third_party/imgui
INCLUDES += -I$(MAKETOP)/third_party/imgui/examples
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

LDFLAGS  += -Wl,--allow-multiple-definition

include ../Makefile.include
