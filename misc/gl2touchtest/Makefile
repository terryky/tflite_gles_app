MAKETOP=../..

include $(MAKETOP)/Makefile.env

TARGET = gl2touchtest

SRCS = 
SRCS += main.c
SRCS += $(MAKETOP)/common/assertgl.c
SRCS += $(MAKETOP)/common/assertegl.c
SRCS += $(MAKETOP)/common/util_egl.c
SRCS += $(MAKETOP)/common/util_shader.c
SRCS += $(MAKETOP)/common/util_matrix.c
SRCS += $(MAKETOP)/common/util_debugstr.c
SRCS += $(MAKETOP)/common/util_pmeter.c
SRCS += $(MAKETOP)/common/util_render2d.c
SRCS += $(MAKETOP)/common/winsys/$(WINSYS_SRC).c

OBJS =
OBJS += $(SRCS:%.c=./%.o)

INCLUDES += -I$(MAKETOP)/common/
INCLUDES += -I$(MAKETOP)/common/winsys/

CFLAGS   +=

LDFLAGS  +=

include $(MAKETOP)/Makefile.include
