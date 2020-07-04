#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = sobelfilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N
	int tempa;
	int tempb;

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	Mat SobelX = Mat::zeros(3, 3, CV_32F);
	Mat SobelY = Mat::zeros(3, 3, CV_32F);

	SobelX.at<float>(0, 0) = -1;
	SobelX.at<float>(0, 2) = 1;
	SobelX.at<float>(1, 0) = -2;
	SobelX.at<float>(1, 2) = 2;
	SobelX.at<float>(2, 0) = -1;
	SobelX.at<float>(2, 2) = 1;

	SobelY.at<float>(0, 0) = -1;
	SobelY.at<float>(0, 1) = -2;
	SobelY.at<float>(0, 2) = -1;
	SobelY.at<float>(2, 0) = 1;
	SobelY.at<float>(2, 1) = 2;
	SobelY.at<float>(2, 2) = 1;


	Mat temp_r = Mat::zeros(row, col, input.type());
	Mat temp_g = Mat::zeros(row, col, input.type());
	Mat temp_b = Mat::zeros(row, col, input.type());
	Mat output = Mat::zeros(row, col, input.type());



	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0;
			float sum1_g = 0;
			float sum1_b = 0;
			float sum2_r = 0;
			float sum2_g = 0;
			float sum2_b = 0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1_r += SobelX.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
					sum1_g += SobelX.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
					sum1_b += SobelX.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
					sum2_r += SobelY.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
					sum2_g += SobelY.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
					sum2_b += SobelY.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
				}
			}
			temp_r.at<C>(i, j) = (G)sqrt(sum1_r*sum1_r + sum2_r * sum2_r);
			temp_g.at<C>(i, j) = (G)sqrt(sum1_g*sum1_g + sum2_g * sum2_g);
			temp_b.at<C>(i, j) = (G)sqrt(sum1_b*sum1_b + sum2_b * sum2_b);

			output.at<C>(i, j) = (C)((temp_r.at<C>(i, j) + temp_g.at<C>(i, j) + temp_b.at<C>(i, j)) / 3);
			
		}
	}

	return output;
}