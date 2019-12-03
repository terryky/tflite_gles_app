#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include "util_image_tga.h"
#include "float16.h"

static void convert_fp32_to_uint8 (int pix_num, int src_ch_num, int ignore_alpha, unsigned char *pdst, float *psrc);
static void convert_fp16_to_uint8 (int pix_num, int src_ch_num, int ignore_alpha, unsigned char *pdst, fp16_t *psrc);

/*
 *  usage:
 *    $ ./bin2tga -f ssbo.bin -w 257 -h 257 -c 3
 */
int 
main (int argc, char *argv[])
{
    char *src_fname = NULL;;
    int src_size, dst_size, pix_num;
    void *src_buf, *dst_buf;
    int src_ch_num = 3;
    int src_width  = 0;
    int src_height = 0;
    int ignore_alpha = 0;
    int src_is_fp16 = 0;
    
    const struct option long_options[] = {
        {"fname",  required_argument, NULL, 'f'},
        {"width",  required_argument, NULL, 'w'},
        {"height", required_argument, NULL, 'h'},
        {"ch_num", required_argument, NULL, 'c'},
        {"ignore_alpha", no_argument, NULL, 'x'},
        {"src_is_fp16",  no_argument, NULL, 'H'},
        {0, 0, 0, 0},
    };

    int c, option_index;
    while ((c = getopt_long (argc, argv, "c:w:h:f:xH",
                             long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 'c': src_ch_num = atoi (optarg); break;
        case 'w': src_width  = atoi (optarg); break;
        case 'h': src_height = atoi (optarg); break;
        case 'f': src_fname  = optarg;        break;
        case 'x': ignore_alpha = 1;           break;
        case 'H': src_is_fp16  = 1;           break;
        case '?':
            return -1;
        }
    }

    if (src_fname == NULL || src_width == 0)
    {
        fprintf (stderr, "usage: bin2tga -f file_name -w width -h height -c channel_num\n");
        return -1;
    }

    fprintf (stderr, "src_fname   : %s\n", src_fname);
    fprintf (stderr, "src_width   : %d\n", src_width);
    fprintf (stderr, "src_height  : %d\n", src_height);
    fprintf (stderr, "src_ch_num  : %d\n", src_ch_num);
    fprintf (stderr, "src_is_fp16 : %d\n", src_is_fp16);
    fprintf (stderr, "ignore_alpha: %d\n", ignore_alpha);
    fprintf (stderr, "\n");

    int fd = open (src_fname, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    struct stat st;
    fstat (fd, &st);
    src_size = st.st_size;
    src_buf  = mmap (0, src_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close (fd);
    if (src_buf == MAP_FAILED)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* number of pixels */
    int byte_per_pix = src_is_fp16 ? 2 : 4;
    pix_num = src_size / byte_per_pix / src_ch_num;
    if (src_height == 0)
        src_height = pix_num / src_width;

    dst_size = pix_num * 4;
    dst_buf = malloc (dst_size);
    if (dst_buf == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* convert (float RGB) ==> (u8 RGBA) */
    if (src_is_fp16)
        convert_fp16_to_uint8 (pix_num, src_ch_num, ignore_alpha, dst_buf, src_buf);
    else
        convert_fp32_to_uint8 (pix_num, src_ch_num, ignore_alpha, dst_buf, src_buf);

    /* emit TGA file */
    char dst_fname[256];
    sprintf (dst_fname, "%s.tga", src_fname);
    save_to_tga_file (dst_fname, dst_buf, src_width, src_height);

    munmap (src_buf, src_size);

    return 0;
}

static float
clamp (float val, float min, float max)
{
    if (val > max)
        val = max;

    if (val < min)
        val = min;

    return val;
}

void
convert_fp32_to_uint8 (int pix_num, int src_ch_num, int ignore_alpha, unsigned char *pdst, float *psrc)
{
    unsigned char uval0 = 0, uval = 0;
    for (int i = 0; i < pix_num; i ++)
    {
        for (int j = 0; j < 4; j ++)
        {
            if (j < src_ch_num)
            {
                float fval = *psrc ++;
                fval = clamp (fval, 0.0f, 1.0f); /* [fixme] need normalization */
                int ival = (int)(fval * 255.0f);
                uval = ival & 0xFF;
                if (j == 0)
                    uval0 = uval;
            }
            else
            {
                uval = uval0;
            }

            if (ignore_alpha && j == 3)
                uval = 0xFF;

            *pdst ++ = uval;
        }
    }
}



void
convert_fp16_to_uint8 (int pix_num, int src_ch_num, int ignore_alpha, unsigned char *pdst, fp16_t *psrc)
{
    unsigned char uval0 = 0, uval = 0;
    for (int i = 0; i < pix_num; i ++)
    {
        for (int j = 0; j < 4; j ++)
        {
            if (j < src_ch_num)
            {
                fp16_t f16val = *psrc ++;
                float fval = half2float (f16val);
                fval = clamp (fval, 0.0f, 1.0f); /* [fixme] need normalization */
                int ival = (int)(fval * 255.0f);
                uval = ival & 0xFF;
                if (j == 0)
                    uval0 = uval;
            }
            else
            {
                uval = uval0;
            }

            if (ignore_alpha && j == 3)
                uval = 0xFF;

            *pdst ++ = uval;
        }
    }
}


