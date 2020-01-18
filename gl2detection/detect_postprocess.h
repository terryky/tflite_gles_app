/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _DETECT_POSTPROCESS_H_
#define _DETECT_POSTPROCESS_H_


struct DetectionBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int   class_id;
};

int init_detect_postprocess (std::string filename);

int
invoke_detection_postprocess (std::vector<DetectionBox> &detection_boxes,  /* [OUT] */
                              const float *boxes_ptr,                      /* [IN ] */
                              const float *_scores_ptr);                   /* [IN ] */

#endif /* _DETECT_POSTPROCESS_H_ */
