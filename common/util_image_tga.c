/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include "util_image_tga.h"

#ifndef UNUSED
#define UNUSED(x) ((void)x)
#endif


typedef enum {
    tga_type_null           = 0,
    tga_type_color_map      = 1,
    tga_type_true_color     = 2,
    tga_type_grayscale      = 3,
    tga_type_rle_color_map  = 9,
    tga_type_rle_true_color = 10,
    tga_type_rle_grayscale  = 11,
} tga_image_type;

typedef struct _tga_colormap_info_t {
    unsigned short      offset;
    unsigned short      length;
    unsigned char       bpp;
} tga_cmap_info_t;

typedef struct _tga_image_info_t {
    unsigned short      x_origin;
    unsigned short      y_origin;
    unsigned short      width;
    unsigned short      height;
    unsigned char       bpp;
    unsigned char       descriptor;
} tga_image_info_t;

typedef struct _tga_header_t {
    unsigned char       id_length;
    unsigned char       cmap_type;
    unsigned char       image_type;
    tga_cmap_info_t     cmap_info;
    tga_image_info_t    image_info;
} tga_header_t;


#define _WRITE(fp, src) {if (fwrite ((&src), sizeof (src), 1, (fp)) != 1) goto write_fail;}

static void *
read_8 (unsigned char *dst, void *p)
{
    unsigned char *q = (unsigned char *)p;
    *dst = *q ++;
    return q;
}

static void *
read_16 (unsigned short *dst, void *p)
{
    unsigned short *q = (unsigned short *)p;
    *dst = *q ++;
    return q;
}

static void *
read_header (u_char *data, tga_header_t *header)
{
    void *p = (void *)data;

    p = read_8  (&header->id_length,  p);
    p = read_8  (&header->cmap_type,  p);
    p = read_8  (&header->image_type, p);

    p = read_16 (&header->cmap_info.offset, p);
    p = read_16 (&header->cmap_info.length, p);
    p = read_8  (&header->cmap_info.bpp,    p);

    p = read_16 (&header->image_info.x_origin,   p);
    p = read_16 (&header->image_info.y_origin,   p);
    p = read_16 (&header->image_info.width,      p);
    p = read_16 (&header->image_info.height,     p);
    p = read_8  (&header->image_info.bpp,        p);
    p = read_8  (&header->image_info.descriptor, p);

    return p;
}


int
open_tga (u_char *data, int size, unsigned int *w, unsigned int *h)
{
    tga_header_t header;

    read_header (data, &header);
    *w = header.image_info.width;
    *h = header.image_info.height;

    return 0;
}

int
open_tga_from_file (char *fname, u_int *w, u_int *h)
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

    ret = open_tga (buf, st.st_size, w, h);
    munmap (buf, st.st_size);

    return ret;
}





int
decode_tga (u_char *src, int src_size, u_char *dst)
{
    tga_header_t header;
    u_char *p;
    int w, h, i;

    p = (u_char *)read_header (src, &header);
    p += header.id_length;
    
    if (header.image_type != tga_type_true_color )
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    w = header.image_info.width;
    h = header.image_info.height;

    switch (header.image_info.bpp)
    {
    case 32:
        for (i = 0; i < w * h; i ++)
        {
            dst[i * 4 + 0] = p[i * 4 + 2];
            dst[i * 4 + 1] = p[i * 4 + 1];
            dst[i * 4 + 2] = p[i * 4 + 0];
            dst[i * 4 + 3] = p[i * 4 + 3];
        }
        break;
    case 24:
        for (i = 0; i < w * h; i ++)
        {
            dst[i * 4 + 0] = p[i * 3 + 2];
            dst[i * 4 + 1] = p[i * 3 + 1];
            dst[i * 4 + 2] = p[i * 3 + 0];
            dst[i * 4 + 3] = 0xFF;
        }
        break;
    default:
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
        break;
    }
 
    return 0;
}

void
decode_tga_from_file (char *fname, u_char *dst)
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

    decode_tga (buf, st.st_size, dst);
    munmap (buf, st.st_size);
}

int 
save_to_tga_file (char *fname, u_char *src, int width, int height)
{
    tga_header_t header = {0};
    FILE *fp;
    size_t i, len;
    u_char *buf;

    fp = fopen (fname, "wb");
    if (fp == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    header.image_type          = tga_type_true_color;
    header.image_info.width    = width;
    header.image_info.height   = height;
    header.image_info.bpp      = 32;

    _WRITE (fp, header.id_length);
    _WRITE (fp, header.cmap_type);
    _WRITE (fp, header.image_type);
    _WRITE (fp, header.cmap_info.offset);
    _WRITE (fp, header.cmap_info.length);
    _WRITE (fp, header.cmap_info.bpp);
    _WRITE (fp, header.image_info.x_origin);
    _WRITE (fp, header.image_info.y_origin);
    _WRITE (fp, header.image_info.width);
    _WRITE (fp, header.image_info.height);
    _WRITE (fp, header.image_info.bpp);
    _WRITE (fp, header.image_info.descriptor);

    len = width * height * 4;
    buf = (u_char *)malloc (len);
    for (i = 0; i < len; i += 4)
    {
        buf[i + 0] = src[i + 2];
        buf[i + 1] = src[i + 1];
        buf[i + 2] = src[i + 0];
        buf[i + 3] = src[i + 3];
    }

    if (fwrite (buf, 1, len, fp) != len)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        free (buf);
        return -1;
    }

    free (buf);
    fclose (fp);

    return 0;
write_fail:
    fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return -1;
}
