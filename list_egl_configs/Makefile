include ../Makefile.env

TARGET = list_egl_configs

SRCS = 
SRCS += main.c
SRCS += ../common/assertegl.c
SRCS += ../common/util_egl.c
SRCS += ../common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I../common/
INCLUDES += -I../common/winsys/

CFLAGS   +=

LDFLAGS  +=

include ../Makefile.include
