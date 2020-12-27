/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_DEBUGSTR_H_
#define _UTIL_DEBUGSTR_H_

#ifdef __cplusplus
extern "C" {
#endif

void init_dbgstr (int win_w, int win_h);
int  draw_dbgstr    (char *str, int x, int y);
int  draw_dbgstr_ex (char *str, int x, int y, float scale, float *col_fg, float *col_bg);

#ifdef __cplusplus
}
#endif
#endif /* _UTIL_DEBUGSTR_H_ */
