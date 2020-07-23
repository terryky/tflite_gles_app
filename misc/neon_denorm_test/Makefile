MAKETOP=../..

include $(MAKETOP)/Makefile.env

TARGET = neon_denorm_test

SRCS =
SRCS += main.c

OBJS =
OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

INCLUDES +=
INCLUDES +=

CFLAGS   += -ffast-math

LDFLAGS  +=
LIBS     +=

include ../../Makefile.include
