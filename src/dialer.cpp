#include "dialer.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <deque>
#include <cassert>

using namespace std;
using namespace cv;

namespace {
const int window_width = 800;
const int window_height = 600;
const char *dialer_window_name = "dialer";
const int moving_average_size = 5;
const int ticks_per_seconds = 25;
const int debounce_delay_ticks = 10;
const int countdown_ticks = 74;

// convert integer to string
string str(int n)
{
	ostringstream ss;
	ss << n;
	return ss.str();
}
}

enum EyeMovement
{
	CENTER = 0,
	LEFT = 1,
	RIGHT = 2,
};

class DialerContext;

class State
{
public:
	virtual void enter(DialerContext *ctx) = 0;
	virtual void exit(DialerContext *ctx) = 0;
	virtual void render(DialerContext *ctx) = 0;
	virtual void eyeMovement(DialerContext *ctx, EyeMovement movement) = 0;
	virtual void tick(DialerContext *ctx) = 0;
	virtual void commitChoice(DialerContext *ctx) = 0;
};

class InputState : public State
{
public:
	virtual void enter(DialerContext *ctx);
	virtual void exit(DialerContext *ctx);
	virtual void render(DialerContext *ctx);
	virtual void eyeMovement(DialerContext *ctx, EyeMovement movement);
	virtual void tick(DialerContext *ctx);
	virtual void commitChoice(DialerContext *ctx);
};

class DialerContext
{
public:
	string input;
	Mat canvas;
	State *state;
	int current_choice_index;
	vector<string> choices;
	deque<float> position_history;
	int wait_ticks;
	int countdown;
	EyeMovement movement;

	DialerContext() : canvas(Mat::zeros(window_height, window_width, CV_8UC3)),
		state(NULL), current_choice_index(0), wait_ticks(0),
		countdown(countdown_ticks)
	{
		setState(new InputState);
		assert(state != NULL);
	}

	~DialerContext()
	{
		delete state;
	}

	void setState(State *new_state)
	{
		assert(new_state != NULL);
		if (new_state == state)
			return;

		if (state)
			state->exit(this);
		new_state->enter(this);

		delete state;
		state = new_state;
	}

	void setChoices(const vector<string>& new_choices)
	{
		choices = new_choices;
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

	void drawTextCentered(const string& s, int x, int y, Scalar color,
			double scale = 1.0)
	{
		const int thickness = 2;
		Size size = getTextSize(s, FONT_HERSHEY_SIMPLEX, scale, thickness, NULL);
		drawText(s, x - size.width / 2, y - size.height / 2, color, scale);
	}

	void show()
	{
		imshow(dialer_window_name, canvas);
	}

	void drawChoices()
	{
		const int center_x = window_width / 2, center_y = window_height / 2;

		// draw the current choice at the center
		drawTextCentered(currentChoice(), center_x, center_y,
				CV_RGB(255, 0, 0), 2.5);

		// draw the previous and next choices
		const string prev_choice = choices[prevChoiceIndex()];
		const string next_choices = choices[nextChoiceIndex()];

		drawTextCentered(prev_choice, center_x - 100, center_y,
				CV_RGB(255, 0, 0), 1.5);
		drawTextCentered(next_choices, center_x + 100, center_y,
				CV_RGB(255, 0, 0), 1.5);
	}

	void drawCountdown() {
		int seconds = countdown / ticks_per_seconds + 1;
		drawTextCentered(str(seconds), window_width / 2, window_height / 4,
				CV_RGB(0, 255, 0), 1.5);
	}

	void drawAll() {
		clear();
		state->render(this);
		show();
	}

	string currentChoice() {
		return choices[current_choice_index];
	}

	void selectNext() {
		current_choice_index = nextChoiceIndex();
	}

	void selectPrev() {
		current_choice_index = prevChoiceIndex();
	}

	int prevChoiceIndex() {
		return (current_choice_index + choices.size() - 1) % choices.size();
	}

	int nextChoiceIndex() {
		return (current_choice_index + 1) % choices.size();
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

	void detectEyeMovement()
	{
		if (wait_ticks > 0) {
			wait_ticks--;
			return;
		}

		float diff = getMovingAverage() - 0.5;
		movement = CENTER;
		if (diff < -0.06)
			movement = LEFT;
		else if (diff > 0.06)
			movement = RIGHT;

		state->eyeMovement(this, movement);

		if (movement != CENTER)
			wait_ticks = debounce_delay_ticks;
	}

	// append s to input
	void inputPush(const string& s)
	{
		input += s;
	}

	// remove the last character from input
	void inputPop()
	{
		if (!input.empty())
			input.erase(input.size() - 1); // remove the last character
	}

	void checkCountdown()
	{
		if (countdown > 0)
			countdown--;
		else {
			countdown = countdown_ticks;
			state->commitChoice(this);
		}
	}

	void tick()
	{
		drawAll();
		checkCountdown();
		detectEyeMovement();
		state->tick(this);
	}

};

// InputState {{{

void InputState::enter(DialerContext *ctx)
{
	vector<string> choices;

	// chocies: 0~9
	for (int i = 0; i <= 9; i++)
		choices.push_back(str(i));
	// delete backward
	choices.push_back("Del");

	ctx->setChoices(choices);
}

void InputState::exit(DialerContext *ctx)
{
}

void InputState::render(DialerContext *ctx)
{
	ctx->drawText(ctx->input, 100, 100, CV_RGB(255, 0, 0));
	ctx->drawChoices();
	ctx->drawCountdown();
}

void InputState::eyeMovement(DialerContext *ctx, EyeMovement movement)
{
	if (movement == LEFT)
		ctx->selectPrev();
	else if (movement == RIGHT)
		ctx->selectNext();

	if (movement != CENTER) {
		ctx->countdown = countdown_ticks;
	}
}

void InputState::tick(DialerContext *ctx)
{
}

void InputState::commitChoice(DialerContext *ctx)
{
	const string& choice = ctx->currentChoice();
	if (choice == "Del")
		ctx->inputPop();
	else
		ctx->inputPush(choice);
}

// }}}

Dialer::Dialer() : ctx(new DialerContext)
{
}

Dialer::~Dialer()
{
	delete ctx;
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
			ctx->selectNext();
			break;
		case 'l':
			ctx->selectPrev();
			break;
	}
}

void Dialer::tick()
{
	ctx->tick();
}

void Dialer::updatePupilPosition(float pupil_left_x, float pupil_left_y,
		float pupil_right_x, float pupil_right_y)
{
	const float position = (pupil_left_x + pupil_right_x) / 2;
	cout << position << endl;
	ctx->position_history.push_back(position);
	while (ctx->position_history.size() > moving_average_size)
		ctx->position_history.pop_front();
}
