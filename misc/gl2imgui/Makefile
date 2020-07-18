MAKETOP=../..

include $(MAKETOP)/Makefile.env

TARGET = gl2imgui

SRCS = 
SRCS += main.c
SRCS += render_imgui.cpp

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
OBJS += $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

# ---------------------
#  for ImGui
# ---------------------
SRCS += $(MAKETOP)/third_party/imgui/imgui.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui_demo.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui_draw.cpp
SRCS += $(MAKETOP)/third_party/imgui/imgui_widgets.cpp
SRCS += $(MAKETOP)/third_party/imgui/examples/imgui_impl_opengl3.cpp

INCLUDES += -I$(MAKETOP)/third_party/imgui
INCLUDES += -I$(MAKETOP)/third_party/imgui/examples

CFLAGS   += -DIMGUI_IMPL_OPENGL_ES2



LDFLAGS  +=

include $(MAKETOP)/Makefile.include
