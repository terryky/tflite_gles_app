/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TOUCH_EVENT_H_
#define TOUCH_EVENT_H_

#ifdef __cplusplus
extern "C" {
#endif

int init_touch_event (int width, int height);

void touch_event_start (int id, int x, int y);
void touch_event_end (int id);
void touch_event_move (int id, int x, int y);

int get_touch_event_matrix (float *mtx);


#ifdef __cplusplus
}
#endif
#endif /* TOUCH_EVENT_H_ */
 