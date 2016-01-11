#ifndef _DIALER_H_
#define _DIALER_H_

class Dialer
{
public:
	Dialer();
	virtual ~Dialer();

	// the app is started
	void start();

	// the app is stopped
	void stop();

	// a key is pressed
	void keypress(int key);

	// this event will be called periodically
	void tick();

	void updatePupilPosition(float pupil_left_x, float pupil_left_y,
			float pupil_right_x, float pupil_right_y);

private:

	// disable copying
	Dialer(const Dialer&);
	Dialer& operator=(const Dialer&);

	class Context;
	Context *ctx;
};

#endif /* _DIALER_H_ */
