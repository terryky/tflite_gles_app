#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <linux/videodev2.h>
#include <poll.h>
#include "util_v4l2.h"
#include "util_drm.h"
#include "util_debug.h"

#define ERRSTR strerror(errno)


static unsigned int
get_capture_device_type (int v4l_fd)
{
    int ret;
    unsigned int caps_flag;
    unsigned int dev_type = 0;
    struct v4l2_capability caps = {0};

    ret = ioctl (v4l_fd, VIDIOC_QUERYCAP, &caps);
    DBG_ASSERT (ret == 0, "VIDIOC_QUERYCAP failed: %s\n", ERRSTR);

    /* if DEVICE_CAPS is enabled, used it */
    if (caps.capabilities & V4L2_CAP_DEVICE_CAPS)
        caps_flag = caps.device_caps;
    else
        caps_flag = caps.capabilities;

    if (caps_flag & V4L2_CAP_VIDEO_CAPTURE)
        dev_type =  V4L2_CAP_VIDEO_CAPTURE;

    if (caps_flag & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
        dev_type =  V4L2_CAP_VIDEO_CAPTURE_MPLANE;

    /* confirm valid format is available */
    if (dev_type)
    {
        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ret = ioctl (v4l_fd, VIDIOC_ENUM_FMT, &fmtdesc);
        if (ret < 0)
            dev_type = 0;
    }

    return dev_type;
}

static unsigned int
get_capture_buftype (unsigned int capture_type)
{
    switch (capture_type)
    {
    case V4L2_CAP_VIDEO_CAPTURE_MPLANE: return V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    case V4L2_CAP_VIDEO_CAPTURE:        return V4L2_BUF_TYPE_VIDEO_CAPTURE;
    default:
        DBG_ASSERT (0, "unknown capture_type");
        return 0;
    }
}

static struct v4l2_format
get_capture_format (capture_dev_t *cap_dev, unsigned int cap_buftype)
{
    int ret;
    struct v4l2_format fmt = {0};

    fmt.type = cap_buftype;
    ret = ioctl (cap_dev->v4l_fd, VIDIOC_G_FMT, &fmt);
    DBG_ASSERT (ret == 0, "VIDIOC_G_FMT failed: %s\n", ERRSTR);

    return fmt;
}

/* ------------------------------------------------------------------------ *
 *  buffer allocation
 * ------------------------------------------------------------------------ */
static int
alloc_buffer_drm (capture_dev_t *cap_dev)
{
    int i;
    struct v4l2_format fmt = cap_dev->stream.format;
    int drm_fd = open_drm();
    int buffer_count = cap_dev->stream.bufcount;
    int buffer_type  = cap_dev->stream.buftype;
    
    int w = fmt.fmt.pix_mp.width;
    int h = fmt.fmt.pix_mp.height;

    for (i = 0; i < buffer_count; i ++)
    {
        drm_fb_t dfb;
        capture_frame_t *cap_frame = &(cap_dev->stream.frames[i]);

        drm_alloc_fb (drm_fd, w, h, DRM_FORMAT_ARGB8888, &dfb);

        cap_frame->vaddr          = dfb.map_buf;
        cap_frame->prime_fd       = dfb.fds[0];
        cap_frame->v4l_buf.index  = i;
        cap_frame->v4l_buf.type   = buffer_type;
        cap_frame->v4l_buf.memory = V4L2_MEMORY_DMABUF;
    }

    return 0;
}

static int
alloc_buffer_mmap (capture_dev_t *cap_dev)
{
    int i, ret;
    int v4l_fd = cap_dev->v4l_fd;
    int buffer_count = cap_dev->stream.bufcount;
    int buffer_type  = cap_dev->stream.buftype;
    
    for (i = 0; i < buffer_count; i ++)
    {
        struct v4l2_buffer buf = {0};
        capture_frame_t *cap_frame = &(cap_dev->stream.frames[i]);

        buf.index  = i; 
        buf.type   = buffer_type;
        buf.memory = V4L2_MEMORY_MMAP;

        ret = ioctl (v4l_fd, VIDIOC_QUERYBUF, &buf);
        DBG_ASSERT (ret == 0, "VIDIOC_QUERYBUF");

        cap_frame->vaddr = mmap (NULL, buf.length, PROT_WRITE|PROT_READ, 
                                 MAP_SHARED, v4l_fd, buf.m.offset);
        
        cap_frame->v4l_buf.index  = i;
        cap_frame->v4l_buf.type   = buffer_type;
        cap_frame->v4l_buf.memory = V4L2_MEMORY_MMAP;

        DBG_ASSERT (cap_frame->vaddr != MAP_FAILED, "mmap");
    }
    return 0;
}

static int
alloc_buffer (capture_dev_t *cap_dev)
{
    capture_stream_t *cap_stream = &(cap_dev->stream);
    int buf_count = cap_stream->bufcount;

    capture_frame_t *cap_frame;
    cap_frame = (capture_frame_t *)malloc (sizeof (capture_frame_t) * buf_count);
    DBG_ASSERT (cap_frame, "alloc failed");

    cap_stream->frames = cap_frame;

    if (cap_stream->memtype == V4L2_MEMORY_DMABUF)
        alloc_buffer_drm (cap_dev);
    else
        alloc_buffer_mmap (cap_dev);

    return 0;
}

static int
init_capture_stream (capture_dev_t *cap_dev, unsigned int buf_memtype, int buf_count)
{
    int ret;
    unsigned int capture_buftype;
    capture_stream_t *cap_stream = &(cap_dev->stream);

    capture_buftype = get_capture_buftype (cap_dev->dev_type);
    DBG_ASSERT (capture_buftype, "not a capture device.\n");

    struct v4l2_requestbuffers rqbufs = {0};
    rqbufs.type   = capture_buftype;
    rqbufs.count  = buf_count;
    rqbufs.memory = buf_memtype;

    ret = ioctl (cap_dev->v4l_fd, VIDIOC_REQBUFS, &rqbufs);
    DBG_ASSERT (ret == 0, "VIDIOC_REQBUFS failed: %s\n", ERRSTR);
    DBG_ASSERT (rqbufs.count >= buf_count, "VIDIOC_REQBUFS failed");

    cap_stream->memtype  = buf_memtype;
    cap_stream->bufcount = buf_count;
    cap_stream->buftype  = capture_buftype;
    cap_stream->format   = get_capture_format (cap_dev, capture_buftype);

    return 0;
}

/* ------------------------------------------------------------------------ *
 *  initialize capture device
 * ------------------------------------------------------------------------ */

int 
v4l2_get_capture_device ()
{
    int i, v4l_fd;
    int dev_id = -1;
    char devname[64];
    unsigned int dev_type;

    for (i = 0; ; i ++)
    {
        snprintf (devname, 64, "/dev/video%d", i);
        v4l_fd = open (devname, O_RDWR | O_CLOEXEC);
        if (v4l_fd < 0)
            break;

        dev_type = get_capture_device_type (v4l_fd);
        if (dev_type)
        {
            dev_id = i;
            close (v4l_fd);
            break;
        }
        close (v4l_fd);
    }
    return dev_id;
}

capture_dev_t *
v4l2_open_capture_device (int devid)
{
    int v4l_fd;
    char devname[64];
    unsigned int dev_type;
    capture_dev_t *cap_dev;

    if (devid < 0)
    {
        devid = v4l2_get_capture_device ();
    }

    if (devid < 0)
        return NULL;

    snprintf (devname, 64, "/dev/video%d", devid);
    v4l_fd = open (devname, O_RDWR | O_CLOEXEC);
    DBG_ASSERT (v4l_fd >= 0, "failed to open %s\n", devname);

    dev_type = get_capture_device_type (v4l_fd);
    DBG_ASSERT (dev_type, "not a capture device.\n");

    cap_dev = (capture_dev_t *)malloc (sizeof (capture_dev_t));
    DBG_ASSERT (cap_dev, "alloc error.\n");

    snprintf (cap_dev->dev_name, sizeof (cap_dev->dev_name), "%s", devname);
    cap_dev->v4l_fd   = v4l_fd;
    cap_dev->dev_type = dev_type;

    init_capture_stream (cap_dev, V4L2_MEMORY_MMAP, 3);
    alloc_buffer (cap_dev);

    return cap_dev;
}


/* ------------------------------------------------------------------------ *
 *  start/stop capture
 * ------------------------------------------------------------------------ */
int
v4l2_start_capture (capture_dev_t *cap_dev)
{
    int i, ret;
    int v4l_fd = cap_dev->v4l_fd;
    capture_stream_t *cap_stream = &cap_dev->stream;

    for (i = 1; i < cap_stream->bufcount; i ++)
    {
        struct v4l2_buffer buf = {0};
        capture_frame_t *cap_frame = &(cap_stream->frames[i]);

        buf.index  = i;
        buf.type   = cap_stream->buftype;
        buf.memory = cap_stream->memtype;

        if (buf.memory == V4L2_MEMORY_DMABUF)
        {
            if (buf.type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
            {
                struct v4l2_plane plane = {0};
                plane.m.fd = cap_frame->prime_fd;
                buf.m.planes = &plane;
                buf.length   = 1;
            }
            else
            {
                buf.m.fd = cap_frame->prime_fd;
            }
        }
        
        ret = ioctl (v4l_fd, VIDIOC_QBUF, &buf);
        DBG_ASSERT (ret == 0, "VIDIOC_QBUF for buffer %d failed: %s\n", i, ERRSTR);
    }

    int type = cap_stream->buftype;
    ret = ioctl (v4l_fd, VIDIOC_STREAMON, &type);
    DBG_ASSERT (ret == 0, "STREAMON failed: %s\n", ERRSTR);


    return 0;
}


/* ------------------------------------------------------------------------ *
 *  acquire/release capture buffer
 * ------------------------------------------------------------------------ */
capture_frame_t *
v4l2_acquire_capture_frame (capture_dev_t *cap_dev)
{
    int ret;
    int v4l_fd = cap_dev->v4l_fd;
    capture_stream_t *cap_stream = &cap_dev->stream;

    struct pollfd fds[1] = {0};
    fds[0].fd     = v4l_fd;
    fds[0].events = POLLIN | POLLERR;

    /* Wait & Dequeue buffer */
    while ((ret = poll (fds, 1, -1)) > 0)
    {
        if (fds[0].revents & POLLIN) 
        {
            struct v4l2_buffer buf = {0};
            buf.type   = cap_stream->buftype;
            buf.memory = cap_stream->memtype;
            ret = ioctl (v4l_fd, VIDIOC_DQBUF, &buf);
            DBG_ASSERT (ret == 0, "VIDIOC_DQBUF failed: %s\n", ERRSTR);

            capture_frame_t *frame = &(cap_stream->frames[buf.index]);
            return frame;
        }
    }

    return 0;
}

int
v4l2_release_capture_frame (capture_dev_t *cap_dev, capture_frame_t *cap_frame)
{
    int ret;
    int v4l_fd = cap_dev->v4l_fd;

    struct v4l2_buffer buf = cap_frame->v4l_buf;
    ret = ioctl (v4l_fd, VIDIOC_QBUF, &buf);
    DBG_ASSERT (ret == 0, "VIDIOC_QBUF failed: %s\n", ERRSTR);

    return 0;
}



/* ------------------------------------------------------------------------ *
 *  utilities
 * ------------------------------------------------------------------------ */
struct v4l2_format
v4l2_get_capture_format (capture_dev_t *cap_dev)
{
    return cap_dev->stream.format;
}

int
v4l2_get_capture_pixelformat (capture_dev_t *cap_dev, unsigned int *pixfmt)
{
    struct v4l2_format infmt = cap_dev->stream.format;
    if (infmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE)
    {
        struct v4l2_pix_format fmt = infmt.fmt.pix;
        *pixfmt = fmt.pixelformat;
    }
    else
    {
        fprintf (stderr, "ERR: %s(%d) not support.\n", __FILE__, __LINE__);
    }
    return 0;
}

int
v4l2_get_capture_wh (capture_dev_t *cap_dev, int *w, int *h)
{
    struct v4l2_format infmt = cap_dev->stream.format;
    if (infmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE)
    {
        struct v4l2_pix_format fmt = infmt.fmt.pix;
        *w = fmt.width;
        *h = fmt.height;
    }
    else
    {
        fprintf (stderr, "ERR: %s(%d) not support.\n", __FILE__, __LINE__);
    }
    return 0;
}


void
v4l2_show_current_capture_settings (capture_dev_t *cap_dev)
{
    unsigned int dev_type = cap_dev->dev_type;
    capture_stream_t *cap_stream = &cap_dev->stream;
    unsigned int capture_buftype = cap_stream->buftype;
    unsigned int capture_memtype = cap_stream->memtype;

    fprintf (stderr, "-------------------------------\n");
    fprintf (stderr, " capture_devie  : %s\n", cap_dev->dev_name);
    fprintf (stderr, " capture_devtype: ");
    if (dev_type == V4L2_CAP_VIDEO_CAPTURE_MPLANE)
        fprintf (stderr, "V4L2_CAP_VIDEO_CAPTURE_MPLANE\n");
    else if (dev_type == V4L2_CAP_VIDEO_CAPTURE)
        fprintf (stderr, "V4L2_CAP_VIDEO_CAPTURE_MPLANE\n");
    else    
        fprintf (stderr, "UNKNOWN\n");

    fprintf (stderr, " capture_buftype: ");
    if (capture_buftype == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        fprintf (stderr, "V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE\n");
    else if (capture_buftype == V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fprintf (stderr, "V4L2_BUF_TYPE_VIDEO_CAPTURE\n");
    else    
        fprintf (stderr, "UNKNOWN\n");
    
    fprintf (stderr, " capture_memtype: ");
    if (capture_memtype == V4L2_MEMORY_DMABUF)
        fprintf (stderr, "V4L2_MEMORY_DMABUF\n");
    else if (capture_memtype == V4L2_MEMORY_MMAP)
        fprintf (stderr, "V4L2_MEMORY_MMAP\n");
    else    
        fprintf (stderr, "UNKNOWN\n");
    
    if (capture_buftype == V4L2_BUF_TYPE_VIDEO_CAPTURE)
    {
        struct v4l2_pix_format fmt = cap_stream->format.fmt.pix;
        fprintf (stderr, " WH(%u, %u), 4CC(%.4s), bpl(%d), size(%d)\n",
            fmt.width, fmt.height, (char *)&fmt.pixelformat,
            fmt.bytesperline, fmt.sizeimage);
    }
    else
    {
        fprintf (stderr, "ERR: %s(%d) not support.\n", __FILE__, __LINE__);
    }
    fprintf (stderr, "-------------------------------\n");
}

