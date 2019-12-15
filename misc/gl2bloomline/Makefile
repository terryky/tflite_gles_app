include ../../Makefile.env

TARGET = gl2bloomline

SRCS = 
SRCS += main.c
SRCS += cube.c
SRCS += filter_normal.c
SRCS += filter_gaussian.c
SRCS += ../../common/assertgl.c
SRCS += ../../common/assertegl.c
SRCS += ../../common/util_egl.c
SRCS += ../../common/util_shader.c
SRCS += ../../common/util_matrix.c
SRCS += ../../common/util_debugstr.c
SRCS += ../../common/util_pmeter.c
SRCS += ../../common/util_render2d.c
SRCS += ../../common/util_render_target.c
SRCS += ../../common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I../../common/
INCLUDES += -I../../common/winsys/

CFLAGS   +=

LDFLAGS  +=
LIBS     +=

include ../../Makefile.include
