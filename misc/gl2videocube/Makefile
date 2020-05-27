include ../../Makefile.env

TARGET = gl2videocube

SRCS = 
SRCS += main.c
SRCS += video_decode.c
SRCS += ../../common/assertgl.c
SRCS += ../../common/assertegl.c
SRCS += ../../common/util_egl.c
SRCS += ../../common/util_shader.c
SRCS += ../../common/util_matrix.c
SRCS += ../../common/util_texture.c
SRCS += ../../common/util_debugstr.c
SRCS += ../../common/util_pmeter.c
SRCS += ../../common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I../../common/
INCLUDES += -I../../common/winsys/
INCLUDES += -I../../third_party/

CFLAGS   +=

LDFLAGS  +=
LIBS     += -lpthread
LIBS     += -ldrm


FFMPEG_LIBS=    libavdevice                        \
                libavformat                        \
                libavfilter                        \
                libavcodec                         \
                libswresample                      \
                libswscale                         \
                libavutil                          \

CFLAGS += $(shell pkg-config --cflags $(FFMPEG_LIBS))
LIBS   += $(shell pkg-config --libs   $(FFMPEG_LIBS)) -lm

include ../../Makefile.include
