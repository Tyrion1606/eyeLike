#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>

void pass_value(float *eye_position, int *mouse_st);
void reset_mouse(void);

void display(void);


void spinDisplay(void);

void init(void);

void reshape(int w, int h);

void mouse(int button, int state, int x, int y);

/*
 *  Request double buffer display mode.
 *  Register mouse input callback functions
 */

