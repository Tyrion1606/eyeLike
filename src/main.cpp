#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For CV_LOAD_IMAGE_COLOR
#include <opencv2/highgui.hpp>   // For CV_WINDOW_NORMAL
#include <vector>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#include "dialer.h"

using namespace cv;

/** Constants **/

/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades
//to your current folder, or change these locations
cv::String face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
Dialer dialer;

int pupil_position_stack[4][5];
int pupil_smooth_position[4];
int pupil_stack_count = 0;
float eye_p[2];
int mouse_click;
float eyetracking_position[8];
float eyetracking_position_right[4];
int program_state= 0;
float tmp_p1[2],tmp_p2[2],tmp_p[2],tmp_pr[2];
typedef struct ThreadArgs {
	int argc;
	const char** argv;
} ThreadArgs;

/**
 * @function main
 */
int main( int argc, const char** argv ) {

	int err;
	ThreadArgs* arg_struct = (ThreadArgs* ) malloc(sizeof(ThreadArgs));
	arg_struct->argc = argc;
	arg_struct->argv = argv;

	//  CvCapture* capture;
	cv::Mat frame;

	// Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){
		printf("--(!)Error loading face cascade,");
		printf("please change face_cascade_name in source code.\n");
		return -1;
	};

	cv::namedWindow(main_window_name,cv::WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 800, 0);
	cv::resizeWindow(main_window_name, 400, 300);
	cv::namedWindow(face_window_name,cv::WINDOW_NORMAL);
	cv::moveWindow(face_window_name, 800, 300);
	cv::resizeWindow(face_window_name, 400, 300);
	//cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
	//cv::moveWindow("Right Eye", 10, 600);
	//cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
	//cv::moveWindow("Left Eye", 10, 800);

	dialer.start();

	createCornerKernels();
	ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
			43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
	VideoCapture capture(0);
	// Read the video stream
	//capture = cvCaptureFromCAM( 1 );
	if( capture.isOpened() ) {
		while( true ) {
			//frame = cvQueryFrame( capture );
			capture.read(frame);
			// mirror it
			cv::flip(frame, frame, 1);
			frame.copyTo(debugImage);

			// Apply the classifier to the frame
			if( !frame.empty() ) {
				detectAndDisplay( frame );
			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			imshow(main_window_name,debugImage);

			int key = cv::waitKey(10);
			if (key != -1) { // -1 means no key is pressed
				if(key == 'q')
					break;
				dialer.keypress(key);
			} else {
				dialer.tick();
			}
		}
	}

	releaseCornerKernels();
	dialer.stop();

	return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
	}
	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth/100.0);
	int eye_region_height = face.width * (kEyePercentHeight/100.0);
	int eye_region_top = face.height * (kEyePercentTop/100.0);
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
			eye_region_top,eye_region_width,eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
			eye_region_top,eye_region_width,eye_region_height);

	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	/*  rectangle(debugFace,leftRightCornerRegion,200);
		rectangle(debugFace,leftLeftCornerRegion,200);
		rectangle(debugFace,rightLeftCornerRegion,200);
		rectangle(debugFace,rightRightCornerRegion,200);
		*///Compute the relative position
	pupil_position_stack[0][pupil_stack_count] = leftPupil.x;
	pupil_position_stack[2][pupil_stack_count] = rightPupil.x;
	//filter out the out of range data
	if( (float) leftPupil.y/leftEyeRegion.height >= 0.23 ||
			(float) leftPupil.y/leftEyeRegion.height <= 0.90	)
		pupil_position_stack[1][pupil_stack_count] = leftPupil.y;
	if( (float) rightPupil.y/rightEyeRegion.height >= 0.23 ||
			(float) rightPupil.y/rightEyeRegion.height <= 0.90)
		pupil_position_stack[3][pupil_stack_count] = rightPupil.y;

	if(pupil_stack_count >= 4){
		pupil_stack_count = 0;
	}
	else {
		pupil_stack_count += 1;
	}
	int loop_count;
	for(loop_count = 0; loop_count<= 3; loop_count++){
		pupil_smooth_position[loop_count] = 0;
	}
	for(loop_count = 0; loop_count<= 4; loop_count++){
		pupil_smooth_position[0] += pupil_position_stack[0][loop_count];
		pupil_smooth_position[1] += pupil_position_stack[1][loop_count];
		pupil_smooth_position[2] += pupil_position_stack[2][loop_count];
		pupil_smooth_position[3] += pupil_position_stack[3][loop_count];
	}
	//printf("\e[1A");
	//printf("\e[K");

	float pupil_left_x  = (float) pupil_smooth_position[0]/(leftEyeRegion.width   * 5);
	float pupil_left_y  = (float) pupil_smooth_position[1]/(leftEyeRegion.height  * 5);
	float pupil_right_x = (float) pupil_smooth_position[2]/(rightEyeRegion.width  * 5);
	float pupil_right_y = (float) pupil_smooth_position[3]/(rightEyeRegion.height * 5);
	//printf("Left pupil: (%.4f,%.4f), Right pupil: (%.4f,%.4f)\n",
	//		pupil_left_x, pupil_left_y, pupil_right_x, pupil_right_y);
	dialer.updatePupilPosition(pupil_left_x, pupil_left_y,
			pupil_right_x, pupil_right_y);

	leftPupil.x  = (int) pupil_smooth_position[0]/5;
	leftPupil.y  = (int) pupil_smooth_position[1]/5;
	rightPupil.x = (int) pupil_smooth_position[2]/5;
	rightPupil.y = (int) pupil_smooth_position[3]/5;
	// program for eyegazing

	//pass_value(eye_p, &mouse_click);

	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	// draw eye centers
	circle(debugFace, rightPupil, 3, 1234);
	circle(debugFace, leftPupil, 3, 1234);

	//-- Find Eye Corners
	if (kEnableEyeCorner) {
		cv::Point2f leftRightCorner =
			findEyeCorner(faceROI(leftRightCornerRegion), true, false);
		leftRightCorner.x += leftRightCornerRegion.x;
		leftRightCorner.y += leftRightCornerRegion.y;
		cv::Point2f leftLeftCorner =
			findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
		leftLeftCorner.x += leftLeftCornerRegion.x;
		leftLeftCorner.y += leftLeftCornerRegion.y;
		cv::Point2f rightLeftCorner =
			findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
		rightLeftCorner.x += rightLeftCornerRegion.x;
		rightLeftCorner.y += rightLeftCornerRegion.y;
		cv::Point2f rightRightCorner =
			findEyeCorner(faceROI(rightRightCornerRegion), false, false);
		rightRightCorner.x += rightRightCornerRegion.x;
		rightRightCorner.y += rightRightCornerRegion.y;
		circle(faceROI, leftRightCorner, 3, 200);
		circle(faceROI, leftLeftCorner, 3, 200);
		circle(faceROI, rightLeftCorner, 3, 200);
		circle(faceROI, rightRightCorner, 3, 200);
	}

	imshow(face_window_name, faceROI);
	//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
	//  cv::Mat destinationROI = debugImage( roi );
	//  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

	cvtColor(frame, input, cv::COLOR_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) {
			cv::Vec3b ycrcb = Mr[x];
			if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
				Or[x] = cv::Vec3b(0,0,0);
			}
		}
	}
	return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
	std::vector<cv::Rect> faces;
	//cv::Mat frame_gray;

	std::vector<cv::Mat> rgbChannels(3);
	cv::split(frame, rgbChannels);
	cv::Mat frame_gray = rgbChannels[2];

	//cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//equalizeHist( frame_gray, frame_gray );
	//cv::pow(frame_gray, CV_64F, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces,
			1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE | cv::CASCADE_FIND_BIGGEST_OBJECT,
			cv::Size(150, 150));

	//  findSkin(debugImage);

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(debugImage, faces[i], 1234);
	}
	//-- Show what you got
	if (faces.size() > 0) {
		findEyes(frame_gray, faces[0]);
	}
}
