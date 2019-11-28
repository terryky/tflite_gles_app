include ../../Makefile.env

TARGET = gl3cs_vbo

SRCS = 
SRCS += main.c
SRCS += ../../common/assertgl.c
SRCS += ../../common/assertegl.c
SRCS += ../../common/util_egl.c
SRCS += ../../common/util_shader.c
SRCS += ../../common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I../../common/
INCLUDES += -I../../common/winsys/

CFLAGS   += -DUSE_GLES_31

LDFLAGS  +=

include ../../Makefile.include
