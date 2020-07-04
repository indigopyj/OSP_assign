#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#include <time.h>
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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
Mat unsharpmask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	clock_t start, finish;
	double duration;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	output = unsharpmask(input, 1, 1, 1, "adjustkernel", 0.5); //Boundary process: zero-paddle, mirroring, adjustkernel
	namedWindow("Unsharp mask", WINDOW_AUTOSIZE);
	imshow("Unsharp mask", output);


	waitKey(0);

	return 0;
}
Mat unsharpmask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {

	Mat gaussian = gaussianfilter(input, n, sigmaT, sigmaS, opt);

	Mat output = (input - k * gaussian) / (1 - k);

	return output;

}

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);


	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with Zero-paddle boundary process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" process:

						Fill the code:
						*/
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
						sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
							sum2 += kernel.at<float>(a + n, b + n);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum2);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);
			}
		}
	}
	return output;
}