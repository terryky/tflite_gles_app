include ../../Makefile.env

TARGET = gl3cs_ssbo_tex

SRCS = 
SRCS += main.c
SRCS += ../../common/assertgl.c
SRCS += ../../common/assertegl.c
SRCS += ../../common/util_egl.c
SRCS += ../../common/util_shader.c
SRCS += ../../common/util_texture.c
SRCS += ../../common/winsys/$(WINSYS_SRC).c

OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

INCLUDES += -I../../common/
INCLUDES += -I../../common/winsys/
INCLUDES += -I../../third_party/

CFLAGS   += -DUSE_GLES_31

LDFLAGS  +=


include ../../Makefile.include
