#include <inttypes.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <sys/poll.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include "util_drm.h"

#define ALIGNN(src_value, align) ((src_value + align-1) & (~(align-1)))

static const char * const crtc_prop_names[] = {
    [WDRM_CRTC_MODE_ID]      = "MODE_ID",
    [WDRM_CRTC_ACTIVE]       = "ACTIVE",
    [WDRM_CRTC_BACKGROUND]   = "background",
};
static const char * const connector_prop_names[] = {
    [WDRM_CONNECTOR_CRTC_ID] = "CRTC_ID",
};

static const char * const plane_prop_names[] = {
    [WDRM_PLANE_TYPE]        = "type",
    [WDRM_PLANE_SRC_X]       = "SRC_X",
    [WDRM_PLANE_SRC_Y]       = "SRC_Y",
    [WDRM_PLANE_SRC_W]       = "SRC_W",
    [WDRM_PLANE_SRC_H]       = "SRC_H",
    [WDRM_PLANE_CRTC_X]      = "CRTC_X",
    [WDRM_PLANE_CRTC_Y]      = "CRTC_Y",
    [WDRM_PLANE_CRTC_W]      = "CRTC_W",
    [WDRM_PLANE_CRTC_H]      = "CRTC_H",
    [WDRM_PLANE_FB_ID]       = "FB_ID",
    [WDRM_PLANE_CRTC_ID]     = "CRTC_ID",
    [WDRM_PLANE_ALPHA]       = "alpha",
    [WDRM_PLANE_COLORKEY]    = "colorkey",
};



/* -------------------------------------------------------------------------- *
 *  DRM FrameBuffer Operation functions.
 * -------------------------------------------------------------------------- */

int 
drm_alloc_fb (int fd, int width, int height, int fourcc, drm_fb_t *dfb)
{
    void *map_buf = NULL;
    struct drm_mode_create_dumb create_arg = {0};
    struct drm_mode_map_dumb    map_arg    = {0};
    struct drm_prime_handle     prime_arg  = {0};
    int i, ret;
    unsigned int alloc_size;

    dfb->width  = width;
    dfb->height = height;
    dfb->fourcc = fourcc;

    for (i = 0; i < CFORMAT_COMPONENT_NUM; i++ )
        dfb->fds[i] = -1;


    switch (dfb->fourcc) {
    case DRM_FORMAT_NV12:     dfb->bpp = 12; dfb->plane_nums = 2; break;
    case DRM_FORMAT_NV16:     dfb->bpp = 16; dfb->plane_nums = 2; break;

    case DRM_FORMAT_RGB565:
    case DRM_FORMAT_YUYV:
    case DRM_FORMAT_UYVY:     dfb->bpp = 16; dfb->plane_nums = 1; break;

    case DRM_FORMAT_ARGB8888: 
    case DRM_FORMAT_XRGB8888: dfb->bpp = 32; dfb->plane_nums = 1; break;
    case DRM_FORMAT_RGB888:
    case DRM_FORMAT_BGR888:   dfb->bpp = 24; dfb->plane_nums = 1; break;

    case DRM_FORMAT_YUV420:
    case DRM_FORMAT_YVU420:   dfb->bpp = 12; dfb->plane_nums = 3; break;

    default:
        fprintf (stderr, "unsupported format 0x%08x\n",  dfb->fourcc);
        return -1;
    }

    if (dfb->plane_nums == 3) 
    {
        if (dfb->bpp == 12) 
        {
            dfb->pitch [0] = ALIGNN(dfb->width, 16);
            dfb->pitch [1] = dfb->pitch [0] / 2;
            dfb->pitch [2] = dfb->pitch [0] / 2;
            dfb->offset[0] = 0;
            dfb->offset[1] = dfb->pitch [0] * dfb->height;
            dfb->offset[2] = dfb->offset[1] + dfb->pitch [1] * dfb->height / 2;
            alloc_size     = dfb->offset[2] + dfb->pitch [2] * dfb->height / 2;
        }
        else
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }
    else if (dfb->plane_nums == 2) 
    {
        dfb->pitch [0] = ALIGNN(dfb->width, 16);
        dfb->offset[0] = 0;

        if (dfb->bpp == 16) 
        {
            dfb->pitch [1] = dfb->pitch [0];
            dfb->offset[1] = dfb->pitch [0] * dfb->height;
            alloc_size     = dfb->offset[1] + dfb->pitch[1] * dfb->height;
        }
        else if (dfb->bpp == 12) 
        {
            dfb->pitch [1] = dfb->pitch [0];
            dfb->offset[1] = dfb->pitch [0] * dfb->height;
            alloc_size     = dfb->offset[1] + dfb->pitch [1] * dfb->height;
        }
        else
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }
    else 
    {
        dfb->pitch [0] = ALIGNN(dfb->width * dfb->bpp / 8, 16);
        dfb->offset[0] = 0;
        alloc_size = dfb->pitch[0] * dfb->height;
    }

    /* Allocate DUMB Buffer --> (create_arg.handle, create_arg.size) */
    create_arg.bpp    = 8;
    create_arg.width  = alloc_size;
    create_arg.height = 1;
    ret = drmIoctl (fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_arg);
    if (ret) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* MMap DUMB Buffer --> cremap_arg.offset */
    map_arg.handle = create_arg.handle;
    ret = drmIoctl (fd, DRM_IOCTL_MODE_MAP_DUMB, &map_arg);
    if (ret) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* mmap() --> map */
    map_buf = mmap (0, create_arg.size, PROT_WRITE|PROT_READ , MAP_SHARED, fd, map_arg.offset);
    if (map_buf == MAP_FAILED) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    prime_arg.handle = create_arg.handle;
    prime_arg.flags  = DRM_CLOEXEC;
    ret = drmIoctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_arg);
    if (ret || prime_arg.fd == -1) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < dfb->plane_nums; i++) 
    {
        dfb->fds[i]    = prime_arg.fd;
        dfb->handle[i] = create_arg.handle;
    }
    dfb->map_buf  = map_buf;
    dfb->map_size = create_arg.size;

    return 0;
}


int 
drm_free_fb (int fd, drm_fb_t *dfb)
{
    struct drm_mode_destroy_dumb arg = {0};
    int ret = 0;

    if (dfb->map_buf)
    {
        munmap (dfb->map_buf, dfb->map_size);
        dfb->map_buf = NULL;

        close (dfb->fds[0]);

        arg.handle = dfb->handle[0];
        ret = drmIoctl (fd, DRM_IOCTL_MODE_DESTROY_DUMB, &arg);
        if (ret)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    return ret;
}

int 
drm_remove_fb (int fd, drm_fb_t *dfb)
{
    if (dfb->fb_id)
    {
        drmModeRmFB (fd, dfb->fb_id);
        dfb->fb_id = 0;
    }

    return 0;
}

int 
drm_add_fb (int fd, drm_fb_t *dfb)
{
    if (drmModeAddFB2 (fd, dfb->width, dfb->height, dfb->fourcc, dfb->handle,
                           dfb->pitch, dfb->offset, &dfb->fb_id, 0)) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return 0;
}


/* -------------------------------------------------------------------------- *
 *  DRM VideoPath Setting Operation functions.
 * -------------------------------------------------------------------------- */

/*
 *  dobj --+-- ddpy[0].crtc_id     : (Encoder & CRTC)
 *         |   ddpy[0].mode_blob_id: (resolution Mode)
 *         |   ddpy[0].conn_id     : (connector)
 *         |   ddpy[0].plane_id    : (plane)
 *         |
 *         +-- ddpy[1].crtc_id     ; (Encoder & CRTC)DE
 *         |   ddpy[1].mode_blob_id: (resolution Mode)
 *         |   ddpy[1].conn_id     : (connector)
 *         |   ddpy[1].plane_id    : (plane)
 *         |
 *         +-- ddpy[2].crtc_id     ; (Encoder & CRTC)
 *             ddpy[2].mode_blob_id: (resolution Mode)
 *             ddpy[2].conn_id     : (connector)
 *             ddpy[2].plane_id    : (plane)
 */

static int 
choose_connector_crtc (int fd, drm_display_t *ddpy, drmModeRes *resources, drmModeConnector *connector)
{
    int i, j;
    int crtc_id = -1;
    int found = 0;
    
    for (i = 0; i < connector->count_encoders; i++) 
    {
        drmModeEncoder *encoder = drmModeGetEncoder (fd, connector->encoders[i]);
        if (encoder == NULL)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }

        uint32_t possible_crtcs = encoder->possible_crtcs;
        for (j = 0; j < resources->count_crtcs; j++) 
        {
            if (possible_crtcs & (1 << j))
            {
                crtc_id = resources->crtcs[j];
                found = 1;
                break;
            }
        }

        drmModeFreeEncoder (encoder);

        if (found)
            break;
    }

    ddpy->crtc_id = crtc_id;

    return found ? 0 : -1;
}

static int 
choose_connector_mode (int fd, drm_display_t *ddpy, drmModeConnector *connector, char *mode_str)
{
    int32_t width  = 0;
    int32_t height = 0;
    int i, ret;
    int configured_flag = 0;
    drmModeModeInfoPtr mode_pref = NULL;
    drmModeModeInfoPtr mode_conf = NULL;
    drmModeModeInfoPtr mode_last = NULL;
    drmModeModeInfoPtr mode_chosen = NULL;

    if (!connector || !connector->count_modes)
        return -1;

    if (mode_str)
    {
        if (sscanf (mode_str, "%dx%d", &width, &height) == 2)
            configured_flag = 1;
    }

    for (i = 0; i < connector->count_modes; i++) 
    {
        if (connector->modes[i].hdisplay == width &&
            connector->modes[i].vdisplay == height)
        {
            if (mode_conf == NULL)
                mode_conf = &connector->modes[i];
        }

        if (connector->modes[i].type & DRM_MODE_TYPE_PREFERRED)
            mode_pref = &connector->modes[i];
    }
    mode_last = &connector->modes[i];

    if (configured_flag)
        mode_chosen = mode_conf ? mode_conf : mode_last;
    else
        mode_chosen = mode_pref ? mode_pref : mode_last;

    ret = drmModeCreatePropertyBlob (fd, mode_chosen, sizeof(*mode_chosen), &ddpy->mode_blob_id);
    if (ret < 0) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return 0;
}


static int 
find_display_by_plane (drm_obj_t *dobj, drmModeRes *resources, drmModePlane *plane)
{
    uint32_t possible_crtcs = plane->possible_crtcs;
    int i, j;

    for (i = 0; i < resources->count_crtcs; i++)
    {
        int crtc_id = -1;
        if (possible_crtcs & (1 << i))
            crtc_id = resources->crtcs[i];

        for (j = 0; j < dobj->display_num; j ++)
        {
            if (dobj->display[j].crtc_id == crtc_id)
                return j;
        }
    }

    return -1;
}

static int
connect_drm_path (drm_obj_t *dobj)
{
    drmModeConnector        *connector;
    drmModeRes              *resources;
    drmModePropertyPtr      property;
    drmModePlaneRes         *plane_res;
    drmModeObjectProperties *props;
    int i, j, k, iddpy, iplane;

    drm_display_t *ddpy = NULL;

    resources = drmModeGetResources (dobj->fd);
    if (resources == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    for (i = 0, iddpy = 0; i < resources->count_connectors; i++) 
    {
        connector = drmModeGetConnector (dobj->fd, resources->connectors[i]);
        if (connector == NULL)
            continue;

        if (connector->connection == DRM_MODE_CONNECTED)
        {
            ddpy = &dobj->display[iddpy];
            iddpy++;

            /* choose display CRTC. ==> (ddpy->crtc_id) */
            if (choose_connector_crtc (dobj->fd, ddpy, resources, connector) < 0)
            {
                fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
                return -1;
            }

            /* choose display resolution MODE. ==> (ddpy->mode_blob_id) */
            if (choose_connector_mode (dobj->fd, ddpy, connector, NULL) < 0)
            {
                fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
                return -1;
            }

            /* setup Connector Property */
            ddpy->con_id = connector->connector_id;
            for (j = 0; j < connector->count_props; j++) 
            {
                property = drmModeGetProperty (dobj->fd, connector->props[j]);
                if (!property)
                    continue;

                for (k = 0; k < WDRM_CONNECTOR__COUNT; k ++) 
                {
                    if (!strcmp(property->name, connector_prop_names[k])) 
                    {
                        ddpy->con_prop_id[k] = property->prop_id;
                        break;
                    }
                }
                drmModeFreeProperty (property);
            }

            /* setup CRTC Property */
            props = drmModeObjectGetProperties (dobj->fd, ddpy->crtc_id, DRM_MODE_OBJECT_CRTC);
            for (j = 0; j < props->count_props; j++) 
            {
                property = drmModeGetProperty (dobj->fd, props->props[j]);
                if (!property)
                    continue;

                for (k = 0; k < WDRM_CRTC__COUNT; k ++) 
                {
                    if (!strcmp(property->name, crtc_prop_names[k])) 
                    {
                        ddpy->crtc_prop_id[k] = property->prop_id;
                        break;
                    }
                }
                drmModeFreeProperty (property);
            }
            drmModeFreeObjectProperties (props);
        }

        drmModeFreeConnector (connector);
        if (iddpy >= MAX_DISPLAY_NUM)
            break;
    }
    dobj->display_num = iddpy;


    /* ------------------------------------------- *
     *  Query Planes and find appropriate Display.
     * ------------------------------------------- */
    plane_res = drmModeGetPlaneResources (dobj->fd);
    if (!plane_res)
    {
        return -1;
    }

    for (i = 0, iplane = 0; i < plane_res->count_planes; i++) 
    {
        drmModePlane *plane;
        drm_plane_t  *dplane;
        int          dpy_idx;

        plane = drmModeGetPlane (dobj->fd, plane_res->planes[i]);
        if (!plane)
            continue;

        dpy_idx = find_display_by_plane (dobj, resources, plane);
        if (dpy_idx < 0)
            continue;

        ddpy   = &dobj->display[dpy_idx];
        dplane = &ddpy->plane[iplane];
        dplane->plane_id = plane->plane_id;
        iplane ++;

        /* setup PLANE Property */
        props = drmModeObjectGetProperties (dobj->fd, plane->plane_id, DRM_MODE_OBJECT_PLANE);
        for (j = 0; j < props->count_props; j++) 
        {
            property = drmModeGetProperty (dobj->fd, props->props[j]);
            if (!property)
                continue;

            for (k = 0; k < WDRM_PLANE__COUNT; k ++) 
            {
                if (!strcmp (property->name, plane_prop_names[k]))
                {
                    dplane->prop_id[k] = property->prop_id;
                    break;
                }
            }
            drmModeFreeProperty (property);
        }

        drmModeFreeObjectProperties (props);
        drmModeFreePlane (plane);

        if (iplane >= MAX_PLANE_NUM)
            break;
    }
    ddpy->plane_num = iplane;

    drmModeFreePlaneResources (plane_res);
    drmModeFreeResources (resources);

    return 0;
}

static void
dump_drm_path (drm_obj_t *dobj)
{
    int i, j;
    
    for (i = 0; i < dobj->display_num; i ++)
    {
        drm_display_t *ddpy = &dobj->display[i];

        fprintf (stderr, "------------ Display[%d/%d] ------------------------\n", i, dobj->display_num);
        fprintf (stderr, " conn_id = %d\n", ddpy->con_id);
        fprintf (stderr, " crtc_id = %d\n", ddpy->crtc_id);
        fprintf (stderr, " mode_id = %d\n", ddpy->mode_blob_id);

        for (j = 0; j < ddpy->plane_num; j ++)
        {
            drm_plane_t *dplane = &ddpy->plane[j];
            fprintf (stderr, " plane[%d/%d].plane_id = %d\n", j, ddpy->plane_num, dplane->plane_id);
        }
    }
}


/* -------------------------------------------------------------------------- *
 *  DRM Open, Close
 * -------------------------------------------------------------------------- */

int
open_drm ()
{
#if defined (DRM_DRIVER_NAME)
    int fd = drmOpen (DRM_DRIVER_NAME, NULL);
#else
    int fd = open ("/dev/dri/card0", O_RDWR | O_CLOEXEC);
#endif

    return fd;
}


int 
drm_initialize (drm_obj_t *dobj)
{
    int fd, ret;

    fd = open_drm ();
    if (fd < 0) 
    {
        fprintf (stderr, "ERR: can't open DRM.\n");
        return -1;
    }

    ret = drmSetClientCap (fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    if (ret) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    ret = drmSetClientCap (fd, DRM_CLIENT_CAP_ATOMIC, 1);
    if (ret) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    ret = drmDropMaster (fd);
    if (ret)
    {
        fprintf (stderr, "ERR: %s(%d): drmDropMaster() failed.\n", __FILE__, __LINE__);
        fprintf (stderr, "     drmDropMaster() requires root privileges.\n");
    }

    dobj->fd = fd;

    ret = connect_drm_path (dobj);
    if (ret) 
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    dump_drm_path (dobj);

    return 0;
}


void 
drm_terminate (drm_obj_t *dobj)
{
    drmClose (dobj->fd);
}


/* -------------------------------------------------------------------------- *
 *  DRM Atomic Operation functions.
 * -------------------------------------------------------------------------- */

int 
drm_atomic_set_mode (drm_obj_t *dobj, int dpy_idx, int enable)
{
    if (dobj->atom == NULL)
    {
        dobj->atom = drmModeAtomicAlloc ();
        if (dobj->atom == NULL) 
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    drm_display_t *ddpy = &dobj->display[dpy_idx];
    int crtc_id = enable ? ddpy->crtc_id : 0;
    int mode_id = enable ? ddpy->mode_blob_id : 0;

    void *atom = dobj->atom;
    drmModeAtomicAddProperty (atom, ddpy->con_id,  ddpy->con_prop_id [WDRM_CONNECTOR_CRTC_ID], crtc_id);
    drmModeAtomicAddProperty (atom, ddpy->crtc_id, ddpy->crtc_prop_id[WDRM_CRTC_MODE_ID],      mode_id);
    drmModeAtomicAddProperty (atom, ddpy->crtc_id, ddpy->crtc_prop_id[WDRM_CRTC_ACTIVE],       enable);
    dobj->atom_flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

    fprintf (stderr, "CONN[%d]--CRTC[%d]--MODE[%d]--ACTIVE[%d]\n", ddpy->con_id, crtc_id, mode_id, enable);

    return 0;
}


int 
drm_atomic_set_plane (drm_obj_t *dobj, drm_fb_t *dfb, int x, int y, int dpy_idx, int plane_idx)
{
    drm_display_t *ddpy   = &dobj->display[dpy_idx];
    drm_plane_t   *dplane = &ddpy->plane  [plane_idx];

    if (dobj->atom == NULL)
    {
        dobj->atom = drmModeAtomicAlloc ();
        if (dobj->atom == NULL) 
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    int fb_id   = dfb ? dfb->fb_id     : 0;
    int fb_x    = dfb ? x              : 0;
    int fb_y    = dfb ? y              : 0;
    int fb_w    = dfb ? dfb->width     : 0;
    int fb_h    = dfb ? dfb->height    : 0;
    int crtc_id = dfb ? ddpy->crtc_id  : 0;
    int plane_id = dplane->plane_id;

    //fprintf (stderr, "plane_id = %d\n", plane_id);
    //fprintf (stderr, "fb_id    = %d\n", fb_id);
    //fprintf (stderr, "crtc_id  = %d\n", crtc_id);
    //fprintf (stderr, "(x, y, w, h) = (%d, %d, %d, %d)\n", fb_x, fb_y, fb_w, fb_x);

    void     *atom = dobj->atom;
    uint32_t *prop = dplane->prop_id;
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_FB_ID],   fb_id);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_CRTC_ID], crtc_id);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_SRC_X],   0);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_SRC_Y],   0);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_SRC_W],   fb_w << 16);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_SRC_H],   fb_h << 16);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_CRTC_X],  fb_x);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_CRTC_Y],  fb_y);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_CRTC_W],  fb_w);
    drmModeAtomicAddProperty (atom, plane_id, prop[WDRM_PLANE_CRTC_H],  fb_h);

    return 0;
}

int
drm_atomic_flush (drm_obj_t *dobj, int block)
{
    if (dobj->atom == NULL)
        return 0;

    if (block == 0)
        dobj->atom_flags |= DRM_MODE_ATOMIC_NONBLOCK | DRM_MODE_PAGE_FLIP_EVENT;

    int ret = drmModeAtomicCommit (dobj->fd, dobj->atom, dobj->atom_flags, NULL);
    if (ret < 0)
    {
        fprintf (stderr, "ERR: failed drmModeAtomicCommit: %s\n", strerror(errno));
    }

    drmModeAtomicFree (dobj->atom);
    dobj->atom       = NULL;
    dobj->atom_flags = 0;

    return ret;
}
