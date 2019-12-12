#ifndef _UTIL_DRM_H
#define _UTIL_DRM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#define MAX_DISPLAY_NUM          3      /* max display num per DRM device. */
#define MAX_PLANE_NUM            4      /* max plane   num per display.    */

enum wdrm_connector_property {
    WDRM_CONNECTOR_CRTC_ID = 0,
    WDRM_CONNECTOR__COUNT
};

enum wdrm_crtc_property {
    WDRM_CRTC_MODE_ID = 0,
    WDRM_CRTC_ACTIVE,
    WDRM_CRTC_BACKGROUND,
    WDRM_CRTC__COUNT
};

enum wdrm_plane_property {
    WDRM_PLANE_TYPE = 0,
    WDRM_PLANE_SRC_X,
    WDRM_PLANE_SRC_Y,
    WDRM_PLANE_SRC_W,
    WDRM_PLANE_SRC_H,
    WDRM_PLANE_CRTC_X,
    WDRM_PLANE_CRTC_Y,
    WDRM_PLANE_CRTC_W,
    WDRM_PLANE_CRTC_H,
    WDRM_PLANE_FB_ID,
    WDRM_PLANE_CRTC_ID,
    WDRM_PLANE_ALPHA,
    WDRM_PLANE_COLORKEY,
    WDRM_PLANE__COUNT
};

#define CFORMAT_COMPONENT_NUM 3 /* Y, U, V */
typedef struct drm_fb_t {
    int width;
    int height;
    int fourcc;
    int bpp;
    int plane_nums;

    uint32_t fb_id;
    uint32_t pitch [CFORMAT_COMPONENT_NUM];
    uint32_t offset[CFORMAT_COMPONENT_NUM];
    uint32_t handle[CFORMAT_COMPONENT_NUM];
    int32_t  fds   [CFORMAT_COMPONENT_NUM];

    void *map_buf;
    int   map_size;
} drm_fb_t;


typedef struct drm_plane_t {
    uint32_t plane_id;
    uint32_t prop_id[WDRM_PLANE__COUNT];
} drm_plane_t;


typedef struct drm_display_t {	
    uint32_t crtc_id;
    uint32_t crtc_prop_id[WDRM_CRTC__COUNT];

    uint32_t con_id;
    uint32_t con_prop_id[WDRM_CONNECTOR__COUNT];

    uint32_t mode_blob_id;

    int plane_num;
    drm_plane_t plane[MAX_PLANE_NUM];
} drm_display_t;


typedef struct drm_obj_t {
    int           fd;

    int           display_num;
    drm_display_t display[MAX_DISPLAY_NUM];

    void          *atom;
    uint32_t      atom_flags;
} drm_obj_t;



/* DRM initialize/terminate */
int  open_drm ();
int  drm_initialize (drm_obj_t *dobj);
void drm_terminate  (drm_obj_t *dobj);

/* DRM Framebuffer operation */
int drm_alloc_fb  (int fd, int width, int height, int fourcc, drm_fb_t *dfb);
int drm_free_fb   (int fd, drm_fb_t *dfb);
int drm_add_fb    (int fd, drm_fb_t *dfb);
int drm_remove_fb (int fd, drm_fb_t *dfb);

/* DRM Atomic operation */
int drm_atomic_set_mode  (drm_obj_t *dobj, int dpy_idx, int enable);
int drm_atomic_set_plane (drm_obj_t *dobj, drm_fb_t *dfb, int x, int y, int dpy_idx, int plane_idx);
int drm_atomic_flush     (drm_obj_t *dobj, int block);

#endif /* _UTIL_DRM_H */

