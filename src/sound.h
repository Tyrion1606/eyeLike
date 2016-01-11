#ifndef _SOUND_H_
#define _SOUND_H_

class Sound
{
public:
	explicit Sound(const char *filename);

	void play() const;
	void playAndWait() const;

private:
	const char *filename_;
};

#endif /* _SOUND_H_ */
