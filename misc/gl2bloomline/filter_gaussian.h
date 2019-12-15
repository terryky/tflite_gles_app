#ifndef _GAUSSIAN_H_
#define _GAUSSIAN_H_

#include "util_render_target.h"

int init_gaussian_blur_filter (float sigma);
int apply_gaussian_filter (render_target_t *dst_fbo, render_target_t *src_fbo);


#endif /* _GAUSSIAN_H_ */
