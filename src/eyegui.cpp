#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include "eyegui.h"

static GLfloat spin = 0.0;
static float xyz_tramslate[3];

float eye_xp;
float eye_yp;
int mouse_status;
static float resize_xy;
void pass_value(float *eye_position, int *mouse_st)
{
	eye_xp = eye_position[0];
	eye_yp = eye_position[1];
	*mouse_st = mouse_status;
	glutPostRedisplay();
}
void reset_mouse(void)
{
	mouse_status = 0;
}
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glPushMatrix();
	glTranslatef(eye_xp,eye_yp,0.0);
	glScalef(resize_xy,1.0,1.0);
	glutSolidSphere(1.0,30,30);
	glPopMatrix();
	glutSwapBuffers();
}


void spinDisplay(void)
{
   glutPostRedisplay();
}

void init(void)
{
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_FLAT);
}

void reshape(int w, int h)
{
	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
	resize_xy = (float) h/w;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-50.0, 50.0, -50.0, 50.0, -10.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

void mouse(int button, int state, int x, int y)
{
   switch (button) {
      case GLUT_LEFT_BUTTON:
         if (state == GLUT_DOWN)
		 {
			mouse_status = 1;
            glutIdleFunc(spinDisplay);
		 }
         break;
      case GLUT_MIDDLE_BUTTON:
         if (state == GLUT_DOWN)
		 {
			mouse_status = 2;
            glutIdleFunc(spinDisplay);
		 }
         break;
      case GLUT_RIGHT_BUTTON:
         if (state == GLUT_DOWN)
		 {
			exit(1);
		 }
         break;
      default:
         break;
   }
}

/*
 *  Request double buffer display mode.
 *  Register mouse input callback functions
 */

int guimain(int argc, char** argv)
{
   glutInit(&argc, argv);
   glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize (1000, 1000);
   //glutInitWindowPosition (100, 100);
   glutCreateWindow (argv[0]);
   glutFullScreen();
   init ();
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutMouseFunc(mouse);
   glutMainLoop();
   return 0;   /* ANSI C requires main to return int. */
}
