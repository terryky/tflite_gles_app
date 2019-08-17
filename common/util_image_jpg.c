/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

#include "util_image_jpg.h"

#ifndef UNUSED
#define UNUSED(x) ((void)x)
#endif

struct my_jpeg_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void
my_error_exit (j_common_ptr cinfo)
{
    struct my_jpeg_error_mgr *mgr = (struct my_jpeg_error_mgr *) cinfo->err;
    longjmp (mgr->setjmp_buffer, 1);
}

static void our_common_init_source (j_decompress_ptr cinfo) { UNUSED (cinfo); }
static void our_common_term_source (j_decompress_ptr cinfo) { UNUSED (cinfo); }

static JOCTET our_memory_buffer[2];

static boolean
our_memory_fill_input_buffer (j_decompress_ptr cinfo)
{
    struct jpeg_source_mgr *src = cinfo->src;

    our_memory_buffer[0] = (JOCTET) 0xFF;
    our_memory_buffer[1] = (JOCTET) JPEG_EOI;

    src->next_input_byte = our_memory_buffer;
    src->bytes_in_buffer = 2;
    return 1;
}

static void
our_memory_skip_input_data (j_decompress_ptr cinfo, long num_bytes)
{
    struct jpeg_source_mgr *src = (struct jpeg_source_mgr *) cinfo->src;

    if (src)
    {
        if ((unsigned long)num_bytes > src->bytes_in_buffer)
            ERREXIT (cinfo, JERR_INPUT_EOF);
        src->bytes_in_buffer -= num_bytes;
        src->next_input_byte += num_bytes;
    }
}


static void
setup_jpeg_src (j_decompress_ptr cinfo, JOCTET *data, unsigned int len)
{
    struct jpeg_source_mgr *src;

    if (cinfo->src == NULL)
    {
        /* First time for this JPEG object?  */
        cinfo->src = (struct jpeg_source_mgr *)
            (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
                                        sizeof (struct jpeg_source_mgr));
      src = (struct jpeg_source_mgr *) cinfo->src;
      src->next_input_byte = data;
    }

    src = (struct jpeg_source_mgr *) cinfo->src;
    src->init_source       = our_common_init_source;
    src->fill_input_buffer = our_memory_fill_input_buffer;
    src->skip_input_data   = our_memory_skip_input_data;
    src->resync_to_restart = jpeg_resync_to_restart; /* Use default method.  */
    src->term_source       = our_common_term_source;
    src->bytes_in_buffer   = len;
    src->next_input_byte   = data;
}

int
open_jpeg (u_char *data, int size, unsigned int *w, unsigned int *h)
{
    struct jpeg_decompress_struct cinfo;
    struct my_jpeg_error_mgr mgr;

    cinfo.err = jpeg_std_error (&mgr.pub);
    mgr.pub.error_exit = my_error_exit;

    if (setjmp (mgr.setjmp_buffer) != 0)
    {
        jpeg_destroy_decompress (&cinfo);
        return -1;
    }

    jpeg_create_decompress (&cinfo);
    setup_jpeg_src (&cinfo, data, size);
    jpeg_read_header (&cinfo, 1);
    *w = cinfo.image_width;
    *h = cinfo.image_height;
    jpeg_destroy_decompress (&cinfo);

    return 0;
}

int
open_jpeg_from_file (char *fname, u_int *w, u_int *h)
{
    int fd;
    struct stat st;
    u_char *buf;
    int ret = 0;

    fd = open (fname, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    
    fstat (fd, &st);
    buf = mmap (0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close (fd);
    if (buf == MAP_FAILED)
        return -2;
    
    ret = open_jpeg (buf, st.st_size, w, h);
    munmap (buf, st.st_size);

    return ret;
}


static void
RGB888toU32ABGR8888 (u_char *src, u_char *dst, int width, int src_bpp)
{
    uint8_t  *tmpbuf = (uint8_t  *) src;
    uint32_t *tmpdst = (uint32_t *) dst;
    int i;

    for (i = 0; i < width; i++)
    {
        *tmpdst = 0xFF      << 24 |
                  tmpbuf[2] << 16 |
                  tmpbuf[1] << 8  |
                  tmpbuf[0] ;
        tmpbuf += src_bpp;
        tmpdst ++;
    }
}


static void
rgb_decode (struct jpeg_decompress_struct *cinfo, u_char *dst)
{
    int row_stride;
    u_char **buffer;

    cinfo->scale_num       = 1;
    cinfo->scale_denom     = 1;
    cinfo->out_color_space = JCS_RGB;

    jpeg_start_decompress (cinfo);
    row_stride = cinfo->output_width * cinfo->output_components;
    buffer = (*cinfo->mem->alloc_sarray)((j_common_ptr) cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo->output_scanline < cinfo->output_height)
    {
        jpeg_read_scanlines (cinfo, buffer, 1);
        RGB888toU32ABGR8888 (*buffer, dst, cinfo->output_width, cinfo->output_components);
        dst += cinfo->output_width * 4;
    }

    jpeg_finish_decompress (cinfo);
}


int
decode_jpeg (u_char *src, int src_size, u_char *dst)
{
    struct jpeg_decompress_struct cinfo;
    struct my_jpeg_error_mgr mgr;

    cinfo.err = jpeg_std_error (&mgr.pub);
    mgr.pub.error_exit = my_error_exit;

    if (setjmp (mgr.setjmp_buffer) != 0)
    {
        jpeg_destroy_decompress (&cinfo);
        return -1;
    }

    jpeg_create_decompress (&cinfo);
    setup_jpeg_src (&cinfo, src, src_size);
    jpeg_read_header (&cinfo, 1);
    rgb_decode (&cinfo, dst);
    jpeg_destroy_decompress (&cinfo);

    return 0;
}

void
decode_jpeg_from_file (char *fname, u_char *dst)
{
    int fd;
    struct stat st;
    u_char *buf;

    fd = open (fname, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return;

    fstat (fd, &st);
    buf = mmap (0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close (fd);
    if (buf == MAP_FAILED)
        return;

    decode_jpeg (buf, st.st_size, dst);
    munmap (buf, st.st_size);
}
