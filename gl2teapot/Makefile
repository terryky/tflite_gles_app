include ../Makefile.env

TARGET = gl2teapot

SRCS = 
SRCS += main.c
SRCS += teapot.c
SRCS += ../common/assertgl.c
SRCS += ../common/assertegl.c
SRCS += ../common/util_egl.c
SRCS += ../common/util_shader.c
SRCS += ../common/util_matrix.c
SRCS += ../common/util_debugstr.c
SRCS += ../common/util_pmeter.c
SRCS += ../common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I../common/
INCLUDES += -I../common/winsys/

CFLAGS   +=

LDFLAGS  +=

include ../Makefile.include
