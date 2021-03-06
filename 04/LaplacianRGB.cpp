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

Mat laplacianfilter(const Mat input);

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
	output = laplacianfilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);



	waitKey(0);

	return 0;
}


Mat laplacianfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N
	int tempa;
	int tempb;

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	Mat Laplacian = Mat::zeros(3, 3, CV_32F);

	Laplacian.at<float>(0, 1) = Laplacian.at<float>(1, 0) = Laplacian.at<float>(1, 2) = Laplacian.at<float>(2, 1) = 1;
	Laplacian.at<float>(1, 1) = -4;


	Mat output = Mat::zeros(row, col, input.type());
	Mat temp_r = Mat::zeros(row, col, input.type());
	Mat temp_g = Mat::zeros(row, col, input.type());
	Mat temp_b = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum_r = 0;
			float sum_g = 0;
			float sum_b = 0;
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
					sum_r += Laplacian.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
					sum_g += Laplacian.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
					sum_b += Laplacian.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
				}
			}
			temp_r.at<C>(i, j) = (C)sqrt(sum_r * sum_r);
			temp_g.at<C>(i, j) = (C)sqrt(sum_g * sum_g);
			temp_b.at<C>(i, j) = (C)sqrt(sum_b * sum_b);

			output.at<C>(i, j) = (C)((temp_r.at<C>(i, j) + temp_g.at<C>(i, j) + temp_b.at<C>(i, j)) / 3);

		}
	}

	return output;
}