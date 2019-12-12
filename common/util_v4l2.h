#ifndef _UTIL_V4L2_H_
#define _UTIL_V4L2_H_

#include <linux/videodev2.h>


typedef struct _capture_frame_t
{
    int     bo_handle;
    int     prime_fd;
    void    *vaddr;
    
    struct v4l2_buffer v4l_buf;
    
} capture_frame_t;

typedef struct _capture_stream_t
{
    unsigned int    memtype;
    unsigned int    buftype;
    int             bufcount;
    capture_frame_t *frames;
    struct v4l2_format format;
} capture_stream_t;


typedef struct _capture_dev_t
{
    int              v4l_fd;
    char             dev_name[64];
    unsigned int     dev_type;
    capture_stream_t stream;
} capture_dev_t;




int              v4l2_get_capture_device ();
capture_dev_t   *v4l2_open_capture_device (int devid);
int              v4l2_start_capture (capture_dev_t *cap_dev);
capture_frame_t *v4l2_acquire_capture_frame (capture_dev_t *cap_dev);
int              v4l2_release_capture_frame (capture_dev_t *cap_dev, capture_frame_t *cap_frame);


int v4l2_get_capture_pixelformat (capture_dev_t *cap_dev, unsigned int *pixfmt);
int v4l2_get_capture_wh (capture_dev_t *cap_dev, int *w, int *h);

void v4l2_show_current_capture_settings (capture_dev_t *cap_dev);

#endif /* _UTIL_V4L2_H_ */
