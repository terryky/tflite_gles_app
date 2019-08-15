/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_WAYLAND_H_
#define UTIL_WAYLAND_H_

#define MOD_SHIFT_MASK    0x01
#define MOD_ALT_MASK      0x02
#define MOD_CONTROL_MASK  0x04

struct Window;
struct Display 
{
	struct wl_display *wlDisplay;
	struct wl_registry *wlRegistry;
	struct wl_compositor *wlCompositor;
	struct wl_shell *wlShell;
	struct wl_seat *wlSeat;
	struct wl_pointer *pointer;
	struct wl_keyboard *keyboard;
	struct xkb_context *xkb_context;
	struct {
		struct xkb_keymap *keymap;
		struct xkb_state *state;
		xkb_mod_mask_t control_mask;
		xkb_mod_mask_t alt_mask;
		xkb_mod_mask_t shift_mask;
	} xkb;
	uint32_t modifiers;
	uint32_t serial;
	struct sigaction sigint;
	struct Window *window;
};

struct Geometry {
    int width, height;
};

struct Window {
	struct Display *display;
	struct wl_egl_window *wlEGLNativeWindow;
	struct wl_surface *wlSurface;
	struct wl_shell_surface *wlShellSurface;
	struct wl_callback *callback;
	int fullscreen, configured, opaque;
	struct Geometry geometry,window_size;
};


typedef void (*NvGlDemoCloseCB)(void);
typedef void (*NvGlDemoResizeCB)(int w, int h);
typedef void (*NvGlDemoKeyCB)(char key, int state);
typedef void (*NvGlDemoPointerCB)(int x, int y);
typedef void (*NvGlDemoButtonCB)(int button, int state);

#endif

