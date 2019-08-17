/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <sys/types.h>
#include <alloca.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>

#include "util_image_png.h"

#ifndef UNUSED
#define UNUSED(x) ((void)x)
#endif

struct my_fp
{
    int size;
    int offset;
    u_char *buf;
};

static inline int
min (int a, int b)
{
    return a <= b ? a : b;
}

static void
my_png_read (png_structp png_ptr, png_bytep data, png_size_t length)
{
    struct my_fp *fp;
    int len;

    if (!png_ptr)
        return;

    fp = png_get_io_ptr (png_ptr);
    len = min (fp->size - fp->offset, length);
    if (len > 0)
        memcpy (data, fp->buf + fp->offset, len);

    fp->offset += len;
}

int
open_png (u_char *data, int size, u_int *w, u_int *h, int *ctype)
{
    struct my_fp fp = {size, 0, data};

    png_structp png_ptr  = png_create_read_struct (PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop   info_ptr = png_create_info_struct (png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct (&png_ptr, 0, 0);
        return -1;
    }
    
    if (setjmp (png_jmpbuf (png_ptr)))
    {
        png_destroy_read_struct (&png_ptr, &info_ptr, 0);
        return -1;
    }

    png_set_read_fn (png_ptr, (void *)&fp, my_png_read);
    png_read_info (png_ptr, info_ptr);

#ifdef PNG_EASY_ACCESS_SUPPORTED
    *w     = png_get_image_width  (png_ptr, info_ptr);
    *h     = png_get_image_height (png_ptr, info_ptr);
    *ctype = png_get_color_type   (png_ptr, info_ptr);
#else
    *w     = info_ptr->width;
    *h     = info_ptr->height;
    *ctype = info_ptr->color_type;
#endif

    png_destroy_read_struct (&png_ptr, &info_ptr, 0);

    return 0;
}

int
open_png_from_file (char *fname, u_int *w, u_int *h, int *ctype)
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

    ret = open_png (buf, st.st_size, w, h, ctype);
    munmap (buf, st.st_size);

    return ret;
}

static void
decode_plte_png (png_structp png_ptr, png_infop info_ptr,
                 u_char **row_ptrs, u_int *plte, int plte_size)
{
    png_color *palette;
    int num_palette;
    u_char *trans, alpha;
    int num_trans;
    png_uint_32 ret;
    int i;

    png_read_image (png_ptr, row_ptrs);

    memset (plte, 0, sizeof(u_int) * plte_size);

    ret = png_get_PLTE (png_ptr, info_ptr, &palette, &num_palette);
    if (ret != PNG_INFO_PLTE)
    {
        fprintf (stderr, "not exist plte\n");
        return;
    }

    if (num_palette > plte_size)
    {
        fprintf (stderr, "plte size over.\n");
        return;
    }
    
#if 1
    png_get_tRNS (png_ptr, info_ptr, &trans, &num_trans, NULL);
#else
    num_trans = (png_get_valid (png_ptr, info_ptr, PNG_INFO_tRNS)) ? info_ptr->num_trans : 0;
    trans = info_ptr->trans;
#endif

    for (i = 0; i < num_palette; i++)
    {
        alpha = (i < num_trans) ? trans[i] : 0xff;

        plte[i] = alpha            << 24 |
                  palette[i].blue  << 16 |
                  palette[i].green <<  8 |
                  palette[i].red         ;
    }
}

static void
plte_to_rgba8888 (u_char **row_ptrs, png_uint_32 width,
                   png_uint_32 height, png_byte bit_depth,
                   u_char *dst, u_int *col)
{
    u_int i, j;

    if (bit_depth == 8)
    {
        for (i = 0; i < height; i++)
        {
            u_char   *s = row_ptrs[i];
            uint32_t *d = (uint32_t *) dst + width * i;
            for (j = 0; j < width; j++, s++, d++)
            {
                *d = col[*s];
            }
        }
    }
    else if (bit_depth == 4)
    {
        for (i = 0; i < height; i++)
        {
            u_char   *s = row_ptrs[i];
            uint32_t *d = (uint32_t *) dst + width * i;
            for (j = 0; j < width; j+=2, s++, d+=2)
            {
                if ((j + 0) < width) {*(d + 0) = col[(*s & 0xf0) >> 4];}
                if ((j + 1) < width) {*(d + 1) = col[(*s & 0x0f) >> 0];}
            }
        }
    }
    else if (bit_depth == 2)
    {
        for (i = 0; i < height; i++)
        {
            u_char   *s = row_ptrs[i];
            uint32_t *d = (uint32_t *) dst + width * i;
            for (j = 0; j < width; j+=4, s++, d+=4)
            {
                if ((j + 0) < width) {*(d + 0)= col[(*s & 0xC0) >> 6];}
                if ((j + 1) < width) {*(d + 1)= col[(*s & 0x30) >> 4];}
                if ((j + 2) < width) {*(d + 2)= col[(*s & 0x0C) >> 2];}
                if ((j + 3) < width) {*(d + 3)= col[(*s & 0x03) >> 0];}
            }
        }
    }
    else if (bit_depth == 1)
    {
        for (i = 0; i < height; i++)
        {
            u_char   *s = row_ptrs[i];
            uint32_t *d = (uint32_t *) dst + width * i;
            for (j = 0; j < width; j+=8, s++, d+=8)
            {
                if ((j + 0) < width) {*(d + 0)= col[(*s & 0x80) >> 7];}
                if ((j + 1) < width) {*(d + 1)= col[(*s & 0x40) >> 6];}
                if ((j + 2) < width) {*(d + 2)= col[(*s & 0x20) >> 5];}
                if ((j + 3) < width) {*(d + 3)= col[(*s & 0x10) >> 4];}
                if ((j + 4) < width) {*(d + 4)= col[(*s & 0x08) >> 3];}
                if ((j + 5) < width) {*(d + 5)= col[(*s & 0x04) >> 2];}
                if ((j + 6) < width) {*(d + 6)= col[(*s & 0x02) >> 1];}
                if ((j + 7) < width) {*(d + 7)= col[(*s & 0x01) >> 0];}
            }
        }
    }
}


int
decode_png (u_char *src, int src_size, u_char *dst)
{
    u_int i;
    u_char *buff;
    u_char **row_ptrs;
    u_int *tmp_plte;
    int plte_size = 256;
    struct my_fp fp = {src_size, 0, src};

    png_structp png_ptr  = png_create_read_struct (PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop   info_ptr = png_create_info_struct (png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct (&png_ptr, 0, 0);
        goto ERROR;
    }

    if (setjmp (png_jmpbuf (png_ptr)))
    {
        png_destroy_read_struct (&png_ptr, &info_ptr, 0);
        goto ERROR;
    }

    png_set_read_fn (png_ptr, (void *)&fp, my_png_read);
    png_read_info (png_ptr, info_ptr);

#ifdef PNG_EASY_ACCESS_SUPPORTED
    u_int    w     = png_get_image_width  (png_ptr, info_ptr);
    u_int    h     = png_get_image_height (png_ptr, info_ptr);
    int      ctype = png_get_color_type   (png_ptr, info_ptr);
    png_byte depth = png_get_bit_depth    (png_ptr, info_ptr);
#else
    u_int    w     = info_ptr->width;
    u_int    h     = info_ptr->height;
    int      ctype = info_ptr->color_type;
    png_byte depth = info_ptr->bit_depth;
#endif

    switch (ctype)
    {
    case PNG_COLOR_TYPE_PALETTE:
        buff = malloc (w * h);
        if (buff == NULL)
        {
            fprintf (stderr, "%s %d malloc error\n", __FILE__, __LINE__);
            goto ERROR;
        }

        pthread_cleanup_push (free, buff);
        row_ptrs = alloca (h * sizeof (u_char *));
        for (i = 0; i < h; i++)
        {
            row_ptrs[i] = buff + w * i;
        }
        tmp_plte = alloca (sizeof(u_int) * plte_size);
        decode_plte_png (png_ptr, info_ptr, row_ptrs, tmp_plte, plte_size);
        plte_to_rgba8888 (row_ptrs, w, h, depth, dst, tmp_plte);

        pthread_cleanup_pop (1);
        break;

    case PNG_COLOR_TYPE_GRAY:
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        buff = malloc (h * w * 2);
        if (buff == NULL)
        {
            fprintf (stderr, "%s %d malloc error\n", __FILE__, __LINE__);
            goto ERROR;
        }

        pthread_cleanup_push (free, buff);
        row_ptrs = alloca (h * sizeof (u_char *));
        for (i = 0; i < h; i++)
        {
            row_ptrs[i] = buff + w * i;
        }

        row_ptrs = alloca (h * sizeof (u_char *));
        for (i = 0; i < h; i++)
        {
            row_ptrs[i] = dst + w * 4 * i;
        }

        png_set_gray_to_rgb (png_ptr);      /* grayscale to RGB */

        if (depth == 16)                    /* 16bit -> 8bit */
            png_set_strip_16 (png_ptr);

        if (depth < 8)                      /* 1/2/4bit to 8bit */
            png_set_expand_gray_1_2_4_to_8 (png_ptr);
        
        if (ctype == PNG_COLOR_TYPE_GRAY_ALPHA)
        {
#if (BYTE_ORDER == BIG_ENDIAN)
            png_set_swap_alpha (png_ptr);
#endif
        }
        else
        {
#if (BYTE_ORDER == BIG_ENDIAN)
            png_set_filler (png_ptr, 0xff, PNG_FILLER_BEFORE);
#else
            png_set_filler (png_ptr, 0xff, PNG_FILLER_AFTER);
#endif
        }
        png_read_image (png_ptr, row_ptrs);

        pthread_cleanup_pop (1);
        break;

    case PNG_COLOR_TYPE_RGB:
        if (depth == 8 || depth == 16)
        {
            row_ptrs = alloca (h * sizeof (u_char *));
            for (i = 0; i < h; i++)
            {
                row_ptrs[i] = dst + w * 4 * i;
            }

#if (BYTE_ORDER == BIG_ENDIAN)
            png_set_bgr (png_ptr);
            png_set_filler (png_ptr, 0xff, PNG_FILLER_BEFORE);
#else
            png_set_filler (png_ptr, 0xff, PNG_FILLER_AFTER);
#endif
            if (depth == 16)
                png_set_strip_16 (png_ptr);
            png_read_image (png_ptr, row_ptrs);
        }
        else
        {
            fprintf (stderr, "not suported png (%d)\n", depth);
            png_destroy_read_struct (&png_ptr, &info_ptr, 0);
            goto ERROR;
        }
        break;

    case PNG_COLOR_TYPE_RGB_ALPHA:
        if (depth == 8 || depth == 16)
        {
            row_ptrs = alloca (h * sizeof (u_char *));
            for (i = 0; i < h; i++)
            {
                row_ptrs[i] = dst + w * 4 * i;
            }

#if (BYTE_ORDER == BIG_ENDIAN)
            png_set_bgr (png_ptr);
            png_set_swap_alpha(png_ptr);
#endif
            if (depth == 16)
                png_set_strip_16 (png_ptr);
            png_read_image (png_ptr, row_ptrs);
        }
        else
        {
            fprintf (stderr, "not suported png (%d)\n", depth);
            png_destroy_read_struct (&png_ptr, &info_ptr, 0);
            goto ERROR;
        }
        break;

    default:
        break;
    }
    png_destroy_read_struct (&png_ptr, &info_ptr, 0);
    return 0;

ERROR:
    return -1;
}

void
decode_png_from_file (char *fname, u_char *dst)
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

    decode_png (buf, st.st_size, dst);
    munmap (buf, st.st_size);
}
