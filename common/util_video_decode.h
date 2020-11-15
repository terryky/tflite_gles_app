/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef VIDEO_DECODE_H_
#define VIDEO_DECODE_H_

int init_video_decode ();
int open_video_file (const char *fname);
int get_video_dimension (int *width, int *height);
int get_video_pixformat (uint32_t *pixformat);
int get_video_buffer (void ** buf);

int start_video_decode ();


#endif
