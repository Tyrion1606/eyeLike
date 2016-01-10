#include "dialer.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <deque>

using namespace std;
using namespace cv;

namespace {
const int window_width = 800;
const int window_height = 600;
const char *dialer_window_name = "dialer";
const int moving_average_size = 5;

// convert integer to string
string str(int n)
{
	ostringstream ss;
	ss << n;
	return ss.str();
}
}

class Dialer::Private
{
public:
	string input;
	Mat canvas;
	int current_choice;
	deque<float> position_history;
	
	Private() : canvas(Mat::zeros(window_height, window_width, CV_8UC3)),
		current_choice(0)
	{
	}

	void clear()
	{
		canvas = CV_RGB(0, 0, 0);
	}

	void drawText(const string& s, int x, int y, Scalar color, double scale = 1.0)
	{
		const int thickness = 2;
		putText(canvas, s, Point(x, y), FONT_HERSHEY_SIMPLEX, scale, color,
				thickness);
	}
	
	void show()
	{
		imshow(dialer_window_name, canvas);
	}

	void prevChoice()
	{
		current_choice = (current_choice + 10 - 1) % 10;
	}

	void nextChoice()
	{
		current_choice = (current_choice + 1) % 10;
	}

	float getMovingAverage()
	{
		if (position_history.empty())
			return 0.5;

		float sum = 0;
		for (int i = 0; i < position_history.size(); i++)
			sum += position_history[i];
		return sum / position_history.size();
	}
};

Dialer::Dialer() : p(new Private)
{
}

Dialer::~Dialer()
{
	delete p;
}

void Dialer::start()
{
	namedWindow(dialer_window_name, CV_WINDOW_NORMAL);
	resizeWindow(dialer_window_name, window_width, window_height);
	moveWindow(dialer_window_name, 0, 0);
}

void Dialer::stop()
{
	destroyWindow(dialer_window_name);
}

void Dialer::keypress(int key)
{
	switch (key)
	{
		case 'h':
			p->prevChoice();
			break;
		case 'l':
			p->nextChoice();
			break;
	}
}

void Dialer::tick()
{
	p->clear();
	p->drawText(p->input, 100, 100, CV_RGB(255, 0, 0));
	p->drawText(str(p->current_choice), window_width/2, window_height/2,
			CV_RGB(255, 0, 0), 2.0);
	p->show();

	if (p->getMovingAverage() < 0.44) {
		p->prevChoice();
	} else if (p->getMovingAverage() > 0.56) {
		p->nextChoice();
	}

}

void Dialer::updatePupilPosition(float pupil_left_x, float pupil_left_y,
		float pupil_right_x, float pupil_right_y)
{
	const float position = (pupil_left_x + pupil_right_x) / 2;
	cout << position << endl;
	p->position_history.push_back(position);
	while (p->position_history.size() > moving_average_size)
		p->position_history.pop_front();
}
