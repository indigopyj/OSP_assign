// opencv_test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter(const Mat input);
Mat Mirroring(const Mat input, int n);


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray, output;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
//	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	//Fill the code using 'Canny' in OpenCV.
	Canny(input_gray, output, 100, 127, 3, false);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", output);

	waitKey(0);

	return 0;
}



