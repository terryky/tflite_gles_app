MAKETOP = ../..

include $(MAKETOP)/Makefile.env

TARGET = gl2tri

SRCS = 
SRCS += main.c
SRCS += $(MAKETOP)/common/assertgl.c
SRCS += $(MAKETOP)/common/assertegl.c
SRCS += $(MAKETOP)/common/util_egl.c
SRCS += $(MAKETOP)/common/util_shader.c
SRCS += $(MAKETOP)/common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I$(MAKETOP)/common/
INCLUDES += -I$(MAKETOP)/common/winsys/

CFLAGS   +=

LDFLAGS  +=

include $(MAKETOP)/Makefile.include
