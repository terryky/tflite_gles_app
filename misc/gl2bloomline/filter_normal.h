#ifndef _FILTER_NORMAL_H_
#define _FILTER_NORMAL_H_

#include "util_render_target.h"

int init_normal_filter ();
int apply_normal_filter (render_target_t *dst_fbo, render_target_t *src_fbo);

#endif /* _FILTER_NORMAL_H_ */
