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
#include <chrono>
#include <thread>
#include <deque>
#include <numeric>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#include "dialer.h"

using namespace cv;
using namespace std;
using namespace std::chrono;


/** Constants **/

/** Function Headers */
void detectAndDisplay( cv::Mat frame );
void findPupil( cv::Mat right_eye );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades
//to your current folder, or change these locations
cv::String face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

cv::String eye_cascade_name = "../../res/haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier eye_cascade;

std::string main_window_name = "Capture - half face";
std::string face_window_name = "Capture - Face";
std::string eye_window_name = "Capture - eye";
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



float fps = 0.0;

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
	if( !eye_cascade.load( eye_cascade_name ) ){
		printf("--(!)Error loading face cascade,");
		printf("please change eye_cascade_name in source code.\n");
		return -1;
	};

	cv::namedWindow(main_window_name,cv::WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 20, 80);
	cv::resizeWindow(main_window_name, 200, 300);

	// createCornerKernels();
	// ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
	// 		43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
	VideoCapture capture(0, cv::CAP_V4L2);

    // Variables to calculate FPS
    auto start = high_resolution_clock::now();
    int frame_count = 0;
    fps = 0.0;
    float duration = 0.0;

	if( capture.isOpened() ) {
        capture.read(frame);
		while( true ) {
			capture.read(frame);

            frame_count++;
            if (frame_count >= 30) {
                auto end = high_resolution_clock::now();
                duration = duration_cast<milliseconds>(end - start).count();
                fps = frame_count / (duration / 1000.0f);
                frame_count = 0;
                start = high_resolution_clock::now();
            }
			cv::flip(frame, frame, 1);

            if( !frame.empty() ) {
				detectAndDisplay( frame );
			}
			int key = cv::waitKey(1);
		}
	}

	// releaseCornerKernels();
	return 0;
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



// Número de frames para a média móvel
const int NUM_FRAMES = 5;
const int NUM_FRAMES_X_AXIS = 5;  // Número de frames para calcular a média de x_axis

// Históricos de detecções
std::deque<cv::Rect> face_history;
std::deque<cv::Rect> eye_history;
std::deque<cv::Vec3i> circle_history;  // Histórico de círculos detectados
std::deque<int> x_axis_history;

cv::Rect getAverageRect(const std::deque<cv::Rect>& rects) {
    int x = 0, y = 0, width = 0, height = 0;
    for (const auto& rect : rects) {
        x += rect.x;
        y += rect.y;
        width += rect.width;
        height += rect.height;
    }
    int n = rects.size();
    return cv::Rect(x / n, y / n, width / n, height / n);
}

int getAverageXAxis(const std::deque<int>& x_axis_values) {
    if (x_axis_values.empty()) return 0;
    int sum = std::accumulate(x_axis_values.begin(), x_axis_values.end(), 0);
    return sum / static_cast<int>(x_axis_values.size());
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay(cv::Mat frame) {
    std::vector<cv::Rect> faces;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces,
        1.3, 2, 0 | cv::CASCADE_SCALE_IMAGE | cv::CASCADE_FIND_BIGGEST_OBJECT,
        cv::Size(200, 200));

    if (!faces.empty()) {
        faces[0].x += faces[0].width/2;
        faces[0].width /= 2;
        faces[0].y += faces[0].height/8;
        faces[0].height /= 2;

        // Atualiza o histórico de faces
        face_history.push_back(faces[0]);
        if (face_history.size() > NUM_FRAMES) {
            face_history.pop_front();
        }

        // Calcula a média móvel da face detectada
        cv::Rect avg_face = getAverageRect(face_history);
        cv::Mat right_face_gray = frame_gray(avg_face);

        // Detecta olhos na metade direita do rosto
        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(right_face_gray, eyes,
            1.2, 2, 0 | cv::CASCADE_SCALE_IMAGE | cv::CASCADE_FIND_BIGGEST_OBJECT,
            cv::Size(25, 25));

        // Atualiza o histórico de olhos e calcula a média móvel
        if (!eyes.empty()) {

            eyes[0].y += eyes[0].height/3;
            eyes[0].height /= 2;


            eye_history.push_back(eyes[0]);
            if (eye_history.size() > NUM_FRAMES) {
                eye_history.pop_front();
            }

            cv::Rect avg_eye = getAverageRect(eye_history);
            cv::Mat right_eye = right_face_gray(avg_eye);

            // findPupil(right_eye);
            // imshow(eye_window_name, right_eye);
            cv::Point rightPupil = findEyeCenter(right_face_gray,avg_eye,"Right Eye");
            // printf("eye-point:[%d,%d] - ", rightPupil.x, rightPupil.y);

            // Calcula e atualiza o histórico de x_axis
            int x_axis = (rightPupil.x - 19)^3;
            x_axis_history.push_back(x_axis);
            if (x_axis_history.size() > NUM_FRAMES_X_AXIS) {
                x_axis_history.pop_front();
            }

            // Calcula a média móvel de x_axis
            int avg_x_axis = getAverageXAxis(x_axis_history);

            // Determina a direção do olhar
            std::string looking = "Center";
            if (avg_x_axis >= 4) {
                looking = "Right";
            } else if (avg_x_axis <= -4) {
                looking = "Left";
            }

            printf("looking(%d): %s\n", avg_x_axis, looking.c_str());
        }

        imshow(main_window_name, right_face_gray);
        

	    cv::Rect area = getAverageRect(face_history);
		area.x += area.width/5;
		area.width /= 3;
		area.y += area.height/2;
		area.height /= 6;
        cv::Mat image = frame_gray(area);
		imshow("teste", image);


        // Print para depuração
        printf("FPS: %.1f - ", fps);
        printf("face width: %d - ", avg_face.width);
        if (!eye_history.empty()) {
            printf("eye width: %d\n", getAverageRect(eye_history).width);
        }
    }
}

cv::Vec3i getAverageCircle(const std::deque<cv::Vec3i>& history) {
    int sum_x = 0, sum_y = 0, sum_radius = 0;
    int count = history.size();

    for (const auto& circle : history) {
        sum_x += circle[0];
        sum_y += circle[1];
        sum_radius += circle[2];
    }

    return cv::Vec3i(sum_x / count, sum_y / count, sum_radius / count);
}

void findPupil(cv::Mat right_eye) {
    // right_eye é uma imagem em preto e branco contendo somente o olho direito

    // Aplicar um filtro Gaussiano para suavizar a imagem
    cv::GaussianBlur(right_eye, right_eye, cv::Size(3, 3), 2);
    right_eye.convertTo(right_eye, -1, 2, -80);

    // Usar a Transformada de Hough para detectar círculos
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(right_eye, circles, cv::HOUGH_GRADIENT, 1, right_eye.cols, 600, 5, 0, 12);

    if (circles.size() > 0) {
        cv::Vec3i c = circles[0];

        // Atualizar o histórico de círculos
        circle_history.push_back(c);
        if (circle_history.size() > NUM_FRAMES) {
            circle_history.pop_front();
        }

        // Calcular a média histórica
        cv::Vec3i avg_circle = getAverageCircle(circle_history);
        cv::Point center = cv::Point(avg_circle[0], avg_circle[1]);

        // Desenhar o círculo central
        cv::circle(right_eye, center, 2, cv::Scalar(255, 255, 255), 1);
        // Desenhar a circunferência do círculo
        cv::circle(right_eye, center, avg_circle[2], cv::Scalar(127, 127, 127), 1);

        // Debug: Print the average circle parameters
        printf("Average Circle: center=(%d, %d), radius=%d\n", center.x, center.y, avg_circle[2]);
    }

    // Exibir a imagem com a pupila detectada
    imshow("Pupil Detection", right_eye);
}