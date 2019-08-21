/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
 
#include <stdio.h>
#include <stdlib.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "assertgl.h"
#include "util_egl.h"

#include "interface/mmal/mmal.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_util_params.h"
#include "interface/khronos/include/EGL/eglext_brcm.h"
#include "RaspiCamControl.h"


static int  s_cam_w = 640;
static int  s_cam_h = 480;

static RASPICAM_CAMERA_PARAMETERS s_cam_param;
static MMAL_COMPONENT_T           *s_cam_component = NULL; // Pointer to the camera component
static MMAL_PORT_T                *s_preview_port  = NULL; // Source port for preview opaque buffers
static MMAL_BUFFER_HEADER_T       *s_mmalbuf       = NULL;
static MMAL_POOL_T                *s_preview_pool  = NULL; // Pool for storing opaque buffer handles
static MMAL_QUEUE_T               *s_preview_queue = NULL; // Queue preview buffers to display in order

static EGLDisplay  s_dpy;
static GLuint      s_texid  = 0;
static EGLImageKHR s_eglimg = EGL_NO_IMAGE_KHR;


static void
preview_output_cb (MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buf)
{
    MMAL_QUEUE_T *preview_queue = (MMAL_QUEUE_T*)port->userdata;

    if (buf->length == 0)
    {
        mmal_buffer_header_release (buf);
    }
    else if (buf->data == NULL)
    {
        mmal_buffer_header_release (buf);
    }
    else
    {
        mmal_queue_put (preview_queue, buf);
    }
}

static int
raspitex_configure_preview_port(MMAL_PORT_T *preview_port)
{
    MMAL_STATUS_T status;

    status = mmal_port_parameter_set_boolean (preview_port,
            MMAL_PARAMETER_ZERO_COPY, MMAL_TRUE);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    preview_port->buffer_num  = preview_port->buffer_num_recommended;
    preview_port->buffer_size = preview_port->buffer_size_recommended;

    /* Pool + queue to hold preview frames */
    s_preview_pool = mmal_port_pool_create(preview_port,
                             preview_port->buffer_num, preview_port->buffer_size);
    if (!s_preview_pool)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* Place filled buffers from the preview port in a queue to render */
    s_preview_queue = mmal_queue_create();
    if (!s_preview_queue)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* Enable preview port callback */
    preview_port->userdata = (struct MMAL_PORT_USERDATA_T*)s_preview_queue;
    status = mmal_port_enable (preview_port, preview_output_cb);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return 0;
}


static MMAL_STATUS_T
create_camera_component ()
{
    MMAL_COMPONENT_T *camera = 0;
    MMAL_ES_FORMAT_T *format;
    MMAL_STATUS_T    status;

    status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    if (!camera->output_num)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    // Enable the camera, and tell it its control callback function
    status = mmal_port_enable(camera->control, default_camera_control_callback);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    //  set up the camera configuration
    {
        MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
        {
         { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) },
         .max_stills_w = s_cam_w,
         .max_stills_h = s_cam_h,
         .stills_yuv422 = 0,
         .one_shot_stills = 1,
         .max_preview_video_w = s_cam_w,
         .max_preview_video_h = s_cam_w,
         .num_preview_video_frames = 3,
         .stills_capture_circular_buffer_height = 0,
         .fast_preview_resume = 0,
         .use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
        };

        mmal_port_parameter_set(camera->control, &cam_config.hdr);
    }

    {
        raspicamcontrol_set_defaults (&s_cam_param);
        raspicamcontrol_set_all_parameters (camera, &s_cam_param);
    }

    // format setting
    s_preview_port = camera->output[0];
    format = s_preview_port->format;
    format->encoding         = MMAL_ENCODING_OPAQUE;
    format->encoding_variant = MMAL_ENCODING_I420;
    format->es->video.width  = VCOS_ALIGN_UP(s_cam_w, 32);
    format->es->video.height = VCOS_ALIGN_UP(s_cam_w, 16);
    format->es->video.crop.x = 0;
    format->es->video.crop.y = 0;
    format->es->video.crop.width  = s_cam_w;
    format->es->video.crop.height = s_cam_w;
    format->es->video.frame_rate.num = 0;
    format->es->video.frame_rate.den = 1;
    status = mmal_port_format_commit (s_preview_port);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* Enable component */
    status = mmal_component_enable (camera);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    status = raspitex_configure_preview_port (s_preview_port);
    if (status != MMAL_SUCCESS)
    {
        fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    s_cam_component = camera;

   return status;

   return status;
}



static int
update_egl_image (MMAL_BUFFER_HEADER_T *buf)
{
    EGLClientBuffer mm_buf = (EGLClientBuffer)buf->data;

    /* destroy previous EGLImage */
    if (s_eglimg != EGL_NO_IMAGE_KHR)
    {
        eglDestroyImageKHR (s_dpy, s_eglimg);
        s_eglimg = EGL_NO_IMAGE_KHR;
    }

    if (s_mmalbuf)
        mmal_buffer_header_release (s_mmalbuf);

    /* create EGLImage */
    glBindTexture (GL_TEXTURE_EXTERNAL_OES, s_texid);
    s_eglimg = eglCreateImageKHR (s_dpy, EGL_NO_CONTEXT, EGL_IMAGE_BRCM_MULTIMEDIA, mm_buf, NULL);
    glEGLImageTargetTexture2DOES (GL_TEXTURE_EXTERNAL_OES, s_eglimg);

    s_mmalbuf = buf;

    return 0;
}


GLuint
create_camera_texture ()
{
    GLuint texid;
    glGenTextures (1, &texid);

    create_camera_component ();

    s_texid = texid;
    s_dpy   = egl_get_display ();
    return texid;
}


int
update_camera_texture ()
{
    MMAL_BUFFER_HEADER_T *buf;
    MMAL_STATUS_T st;
    int rc = 0;

    /* Send empty buffers to camera preview port */
    while ((buf = mmal_queue_get (s_preview_pool->queue)) != NULL)
    {
        st = mmal_port_send_buffer (s_preview_port, buf);
        if (st != MMAL_SUCCESS)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    /* Acquire buffers */
    while ((buf = mmal_queue_get (s_preview_queue)) != NULL)
    {
        rc = update_egl_image (buf);
        if (rc != 0)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return rc;
        }
    }

    return 0;
}


