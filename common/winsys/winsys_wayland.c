/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <signal.h>
#if defined (USE_XKBCOMMON)
#include <xkbcommon/xkbcommon.h>
#endif
#include "wayland-client.h"
#include "wayland-egl.h"
#include "winsys_wayland.h"

#define UNUSED(x) (void)(x)

struct Display s_display;
struct Window  s_window;

static void
handle_ping(void *data, struct wl_shell_surface *wlShellSurface,
        uint32_t serial)
{
    UNUSED (data);

    wl_shell_surface_pong(wlShellSurface, serial);
}

static void
handle_configure(void *data, struct wl_shell_surface *shell_surface,
             uint32_t edges, int32_t width, int32_t height)
{
    struct Window *window = data;
    UNUSED (data);
    UNUSED (shell_surface);
    UNUSED (edges);

    if (window->wlEGLNativeWindow) {
        wl_egl_window_resize(window->wlEGLNativeWindow, width, height, 0, 0);
    }

    window->geometry.width = width;
    window->geometry.height = height;

    if (!window->fullscreen) {
        window->window_size = window->geometry;
    }
}

static const struct wl_shell_surface_listener shell_surface_listener =
{
    handle_ping,
    handle_configure,
    NULL
};

static void
configure_callback(void *data, struct wl_callback *callback, uint32_t time)
{
    struct Window *window = data;
    UNUSED (time);
    
    wl_callback_destroy(callback);

    window->configured = 1;
}

static struct wl_callback_listener configure_callback_listener =
{
    configure_callback,
};

static void
toggle_fullscreen(struct Window *window, int fullscreen)
{

    struct wl_callback *callback;

    window->fullscreen = fullscreen;
    window->configured = 0;

    if (fullscreen) {
        wl_shell_surface_set_fullscreen(
            window->wlShellSurface,
            WL_SHELL_SURFACE_FULLSCREEN_METHOD_DEFAULT,0, NULL);
    } else {
        wl_shell_surface_set_toplevel(window->wlShellSurface);
        handle_configure(window, window->wlShellSurface, 0,
            window->window_size.width,
            window->window_size.height);
    }

    callback = wl_display_sync(window->display->wlDisplay);
    wl_callback_add_listener(callback, &configure_callback_listener,
        window);

}

static void
pointer_handle_enter(void *data, struct wl_pointer *pointer,
                     uint32_t serial, struct wl_surface *surface,
                     wl_fixed_t sx, wl_fixed_t sy)
{
    UNUSED (data);
    UNUSED (pointer);
    UNUSED (serial);
    UNUSED (surface);
    UNUSED (sx);
    UNUSED (sy);
}

static void
pointer_handle_leave(void *data, struct wl_pointer *pointer,
                     uint32_t serial, struct wl_surface *surface)
{
    UNUSED (data);
    UNUSED (pointer);
    UNUSED (serial);
    UNUSED (surface);
}

static void
pointer_handle_motion(void *data, struct wl_pointer *pointer,
                      uint32_t time, wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    UNUSED (data);
    UNUSED (pointer);
    UNUSED (time);
}

static void
pointer_handle_button(void *data, struct wl_pointer *wl_pointer,
                      uint32_t serial, uint32_t time, uint32_t button,
                      uint32_t state)
{
    UNUSED (data);
    UNUSED (wl_pointer);
    UNUSED (serial);
    UNUSED (time);
    UNUSED (button);
    UNUSED (state);
#if 0
    if (buttonCB) {
        buttonCB((button == BTN_LEFT) ? 1 : 0,
            (state == WL_POINTER_BUTTON_STATE_PRESSED) ? 1 : 0);
    }
#endif
}

static void
pointer_handle_axis(void *data, struct wl_pointer *wl_pointer,
                    uint32_t time, uint32_t axis, wl_fixed_t value)
{
    UNUSED (data);
    UNUSED (wl_pointer);
    UNUSED (time);
    UNUSED (axis);
    UNUSED (value);
}

static const struct wl_pointer_listener pointer_listener =
{
    pointer_handle_enter,
    pointer_handle_leave,
    pointer_handle_motion,
    pointer_handle_button,
    pointer_handle_axis,
};

static void
keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
                       uint32_t format, int fd, uint32_t size)
{
#if defined (USE_XKBCOMMON)
    struct Display *input = data;
    struct xkb_keymap *keymap;
    struct xkb_state *state;
    char *map_str;
    UNUSED (keyboard);

    if (!data) {
        close(fd);
        return;
    }

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
        close(fd);
        return;
    }

    map_str = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map_str == MAP_FAILED) {
        close(fd);
        return;
    }

    keymap = xkb_map_new_from_string(input->xkb_context,
                map_str, XKB_KEYMAP_FORMAT_TEXT_V1, 0);
    munmap(map_str, size);
    close(fd);

    if (!keymap) {
        fprintf (stderr, "failed to compile keymap\n");
        return;
    }

    state = xkb_state_new(keymap);
    if (!state) {
        fprintf (stderr, "failed to create XKB state\n");
        xkb_map_unref(keymap);
        return;
    }

    xkb_keymap_unref(input->xkb.keymap);
    xkb_state_unref(input->xkb.state);
    input->xkb.keymap = keymap;
    input->xkb.state = state;

    input->xkb.control_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Control");
    input->xkb.alt_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Mod1");
    input->xkb.shift_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Shift");
#endif
}

static void
keyboard_handle_enter(void *data, struct wl_keyboard *keyboard,
                      uint32_t serial, struct wl_surface *surface,
                      struct wl_array *keys)
{
    UNUSED (data);
    UNUSED (keyboard);
    UNUSED (serial);
    UNUSED (surface);
    UNUSED (keys);
}

static void
keyboard_handle_leave(void *data, struct wl_keyboard *keyboard,
                      uint32_t serial, struct wl_surface *surface)
{
    UNUSED (data);
    UNUSED (keyboard);
    UNUSED (serial);
    UNUSED (surface);
}

static void
keyboard_handle_key(void *data, struct wl_keyboard *keyboard,
                    uint32_t serial, uint32_t time, uint32_t key,
                    uint32_t state)
{
#if defined (USE_XKBCOMMON)
    struct Display *input = data;
    uint32_t code, num_syms;
    const xkb_keysym_t *syms;
    xkb_keysym_t sym;
    UNUSED (keyboard);
    UNUSED (time);
    
    input->serial = serial;
    code = key + 8;
    if (!input->xkb.state) {
        return;
    }

    num_syms = xkb_key_get_syms(input->xkb.state, code, &syms);

    sym = XKB_KEY_NoSymbol;
    if (num_syms == 1) {
        sym = syms[0];
    }

    if (keyCB) {
        if (sym) {
            char buf[16];
            xkb_keysym_to_utf8(sym, &buf[0], 16);
            keyCB(buf[0], (state == WL_KEYBOARD_KEY_STATE_PRESSED)? 1 : 0);
        }
    }
#endif
}

static void
keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
                          uint32_t serial, uint32_t mods_depressed,
                          uint32_t mods_latched, uint32_t mods_locked,
                          uint32_t group)
{
#if defined (USE_XKBCOMMON)
    struct Display *input = data;
    xkb_mod_mask_t mask;
    UNUSED (keyboard);
    UNUSED (serial);
    
    /* If we're not using a keymap, then we don't handle PC-style modifiers */
    if (!input->xkb.keymap) {
        return;
    }

    xkb_state_update_mask(input->xkb.state, mods_depressed, mods_latched,
        mods_locked, 0, 0, group);

    mask = xkb_state_serialize_mods(input->xkb.state,
        XKB_STATE_DEPRESSED |
        XKB_STATE_LATCHED);
    input->modifiers = 0;

    if (mask & input->xkb.control_mask) {
        input->modifiers |= MOD_CONTROL_MASK;
    }
    if (mask & input->xkb.alt_mask) {
        input->modifiers |= MOD_ALT_MASK;
    }
    if (mask & input->xkb.shift_mask) {
        input->modifiers |= MOD_SHIFT_MASK;
    }
#endif
}

static const struct wl_keyboard_listener keyboard_listener =
{
    keyboard_handle_keymap,
    keyboard_handle_enter,
    keyboard_handle_leave,
    keyboard_handle_key,
    keyboard_handle_modifiers,
    NULL
};

static void
seat_handle_capabilities(void *data, struct wl_seat *seat,
                         enum wl_seat_capability caps)
{
    struct Display *d = data;

    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !d->pointer) {
        d->pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(d->pointer, &pointer_listener, d);
    }

    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !d->keyboard) {
        d->keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(d->keyboard, &keyboard_listener, d);
    }
}

static const struct wl_seat_listener seat_listener =
{
    seat_handle_capabilities,
    NULL
};

// Registry handling static function
static void
registry_handle_global(void *data, struct wl_registry *registry,
               uint32_t name, const char *interface, uint32_t version)
{
    struct Display *d = data;
    UNUSED (version);

    if (strcmp(interface, "wl_compositor") == 0) {
        d->wlCompositor = wl_registry_bind(registry, name,
                        &wl_compositor_interface, 1);
    } else if (strcmp(interface, "wl_shell") == 0) {
        d->wlShell = wl_registry_bind(registry, name,
                        &wl_shell_interface, 1);
    } else if (strcmp(interface, "wl_seat") == 0) {
        d->wlSeat = wl_registry_bind(registry, name,
                        &wl_seat_interface, 1);
        wl_seat_add_listener(d->wlSeat, &seat_listener, d);
    }
}

static void
registry_handle_global_remove(void *data, struct wl_registry *registry,
                  uint32_t name)
{
    UNUSED (data);
    UNUSED (registry);
    UNUSED (name);
}

static const struct wl_registry_listener registry_listener = {
    registry_handle_global,
    registry_handle_global_remove
};





void *
winsys_init_native_display (void)
{
    memset (&s_display, 0, sizeof (s_display));


    s_display.wlDisplay = wl_display_connect(NULL);
    if (s_display.wlDisplay == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }

#if defined (USE_XKBCOMMON)
    s_display.xkb_context = xkb_context_new(0);
    if (s_display.xkb_context == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }
#endif

    s_display.wlRegistry = wl_display_get_registry(s_display.wlDisplay);
    if (s_display.wlRegistry == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }

    wl_registry_add_listener(s_display.wlRegistry, &registry_listener, &s_display);

    wl_display_dispatch(s_display.wlDisplay);

    return s_display.wlDisplay;;
}


void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    UNUSED (dpy);
    memset (&s_window, 0, sizeof (s_window));

    if (!s_display.wlCompositor || !s_display.wlShell)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }
    
    s_window.wlSurface = wl_compositor_create_surface(s_display.wlCompositor);
    if (s_window.wlSurface == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }

    s_window.wlShellSurface =
       wl_shell_get_shell_surface(s_display.wlShell, s_window.wlSurface);
    if (s_window.wlShellSurface == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }
    
    wl_shell_surface_add_listener(s_window.wlShellSurface, &shell_surface_listener, &s_window);

    s_window.window_size.width  = win_w;
    s_window.window_size.height = win_h;
    s_window.display = &s_display;
    toggle_fullscreen(&s_window, 0);    
    
    s_window.wlEGLNativeWindow =  wl_egl_window_create(s_window.wlSurface, win_w, win_h);
    if (s_window.wlEGLNativeWindow == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }

    return s_window.wlEGLNativeWindow;
}


int 
winsys_swap()
{
  return 0;
}

void *
winsys_create_native_pixmap (int width, int height)
{
  return NULL;
}

