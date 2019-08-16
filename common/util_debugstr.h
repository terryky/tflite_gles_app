/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_DEBUGSTR_H_
#define _UTIL_DEBUGSTR_H_



void dbgstr_initialize (int w, int h);
int  dbgstr_draw    (char *str, int x, int y);
int  dbgstr_draw_ex (char *str, int x, int y, float scale, float *col_fg, float *col_bg);

#endif /* _UTIL_DEBUGSTR_H_ */
