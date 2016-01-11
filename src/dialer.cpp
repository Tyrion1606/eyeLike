#include "dialer.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <deque>
#include <cassert>
#include "sound.h"

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
const float eye_movement_threashold = 0.06;

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
	virtual void enter(DialerContext *ctx) {};
	virtual void exit(DialerContext *ctx) {};
	virtual void render(DialerContext *ctx) {};
	virtual void eyeMovement(DialerContext *ctx, EyeMovement movement) {};
	virtual void tick(DialerContext *ctx) {};
	virtual void commitChoice(DialerContext *ctx) {};
};

class InputState : public State
{
public:
	InputState();
	virtual void enter(DialerContext *ctx);
	virtual void render(DialerContext *ctx);
	virtual void eyeMovement(DialerContext *ctx, EyeMovement movement);
	virtual void commitChoice(DialerContext *ctx);
protected:
	Sound sound_select, sound_change;
};

class WaitState : public State
{
public:
	virtual void enter(DialerContext *ctx);
	virtual void render(DialerContext *ctx);
	virtual void eyeMovement(DialerContext *ctx, EyeMovement movement);
	virtual void tick(DialerContext *ctx);
private:
	EyeMovement prev_movement;
	int points;
};

// TODO: extract common base case with InputState
class ConfirmState : public InputState
{
public:
	virtual void enter(DialerContext *ctx);
	virtual void render(DialerContext *ctx);
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
		setState(new WaitState);
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
		current_choice_index = 0;
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
		if (diff < -eye_movement_threashold)
			movement = LEFT;
		else if (diff > eye_movement_threashold)
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

InputState::InputState()
	: sound_select("select.ogg"), sound_change("change.ogg")
{
}

void InputState::enter(DialerContext *ctx)
{
	vector<string> choices;

	// chocies: 0~9
	for (int i = 0; i <= 9; i++)
		choices.push_back(str(i));
	// delete backward
	choices.push_back("Del");
	choices.push_back("Call");

	ctx->setChoices(choices);
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
		sound_change.play();
	}
}

void InputState::commitChoice(DialerContext *ctx)
{
	sound_select.play();

	const string& choice = ctx->currentChoice();
	if (choice == "Del")
		ctx->inputPop();
	else if (choice == "Call")
		ctx->setState(new ConfirmState);
	else
		ctx->inputPush(choice);
}

// }}}

// WaitState {{{

void WaitState::enter(DialerContext *ctx)
{
	prev_movement = CENTER;
	points = 0;
}

void WaitState::render(DialerContext *ctx)
{
	ctx->drawTextCentered("Quickly look left and right 5 times to start",
			window_width/2, window_height/2,
			CV_RGB(255, 255, 255));
}

void WaitState::eyeMovement(DialerContext *ctx, EyeMovement movement)
{
	// We are only interested in left and right in this state.
	if (movement == CENTER)
		return;

	if (prev_movement != movement) // transition from left to right or vice versa
		points += 1.5 * ticks_per_seconds;

	if (points >= ticks_per_seconds * 5) {
		ctx->setState(new InputState);
		return;
	}

	prev_movement = movement;
}

void WaitState::tick(DialerContext *ctx)
{
	if (points > 0)
		points--;
}

// }}}

// ConfirmState {{{

void ConfirmState::enter(DialerContext *ctx)
{
	vector<string> choices;
	choices.push_back("No");
	choices.push_back("Yes");
	ctx->setChoices(choices);
}

void ConfirmState::render(DialerContext *ctx)
{
	string msg("Do you want to call ");
	msg += ctx->input;

	ctx->drawTextCentered(msg, window_width/2, window_height/2 + 100,
			CV_RGB(255, 255, 0));
	ctx->drawChoices();
	ctx->drawCountdown();
}

void ConfirmState::commitChoice(DialerContext *ctx)
{
	sound_select.play();
	const string choice = ctx->currentChoice();
	if (choice == "Yes") {
		// make phone call
	}
	ctx->setState(new WaitState);
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
	ctx->position_history.push_back(position);
	while (ctx->position_history.size() > moving_average_size)
		ctx->position_history.pop_front();
}
