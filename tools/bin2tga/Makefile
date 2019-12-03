include ../../Makefile.env

TARGET = bin2tga

SRCS = 
SRCS += main.c
SRCS += ../../common/util_image_tga.c

OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

INCLUDES += -I../../common/


LDFLAGS  +=


include ../../Makefile.include
