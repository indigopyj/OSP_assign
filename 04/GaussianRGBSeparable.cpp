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

Mat gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

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

	start = clock();
	output = gaussianfilter_sep(input, 1, 1, 1, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel
	finish = clock();
	namedWindow("Gaussian Filter - separable", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter - separable", output);

	duration = (double)(finish - start);
	printf("Gaussian in separable : %f sec", duration);

	waitKey(0);

	return 0;
}


Mat gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {


	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	//float kernelvalue;

		// Initialiazing Kernel Matrix 
	Mat kernel_s = Mat::zeros(kernel_size, 1, CV_32F);
	Mat kernel_t = Mat::zeros(kernel_size, 1, CV_32F);


	float denom_s = 0.0;
	float denom_t = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)

		float value_s = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		float value_t = exp(-(pow(a, 2) / (2 * pow(sigmaT, 2))));
		kernel_s.at<float>(a + n) = value_s;
		kernel_t.at<float>(a + n) = value_t;

		denom_s += value_s;
		denom_t += value_t;

	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)

		kernel_s.at<float>(a + n) /= denom_s;
		kernel_t.at<float>(a + n) /= denom_t;

	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				float sum_s_r = 0.0;
				float sum_s_g = 0.0;
				float sum_s_b = 0.0;
				for (int a = -n; a <= n; a++) {
					float sum_t_r = 0.0;
					float sum_t_g = 0.0;
					float sum_t_b = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum_t_r += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum_t_g += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum_t_b += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[2]);
						}
					}
					sum_s_r += kernel_s.at<float>(a + n) * sum_t_r;
					sum_s_g += kernel_s.at<float>(a + n) * sum_t_g;
					sum_s_b += kernel_s.at<float>(a + n) * sum_t_b;


				}
				output.at<C>(i, j)[0] = (G)sum_s_r;
				output.at<C>(i, j)[1] = (G)sum_s_g;
				output.at<C>(i, j)[2] = (G)sum_s_b;

			}

			else if (!strcmp(opt, "mirroring")) {
				float sum_s_r = 0.0;
				float sum_s_g = 0.0;
				float sum_s_b = 0.0;
				for (int a = -n; a <= n; a++) {
					float sum_t_r = 0.0;
					float sum_t_g = 0.0;
					float sum_t_b = 0.0;
					for (int b = -n; b <= n; b++) {

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
						sum_t_r += kernel_t.at<float>(b + n) * (float)(input.at<C>(tempa, tempb)[0]);
						sum_t_g += kernel_t.at<float>(b + n) * (float)(input.at<C>(tempa, tempb)[1]);
						sum_t_b += kernel_t.at<float>(b + n) * (float)(input.at<C>(tempa, tempb)[2]);


					}
					sum_s_r += kernel_s.at<float>(a + n) * sum_t_r;
					sum_s_g += kernel_s.at<float>(a + n) * sum_t_g;
					sum_s_b += kernel_s.at<float>(a + n) * sum_t_b;

				}
				output.at<C>(i, j)[0] = (G)sum_s_r;
				output.at<C>(i, j)[1] = (G)sum_s_g;
				output.at<C>(i, j)[2] = (G)sum_s_b;

			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum_s_r = 0.0;
				float sum2_s_r = 0.0;
				float sum_s_g = 0.0;
				float sum2_s_g = 0.0;
				float sum_s_b = 0.0;
				float sum2_s_b = 0.0;

				for (int a = -n; a <= n; a++) {
					float sum_t_r = 0.0;
					float sum_t_g = 0.0;
					float sum_t_b = 0.0;
					float sum2_t_r = 0.0;
					float sum2_t_g = 0.0;
					float sum2_t_b = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum_t_r += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum_t_g += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum_t_b += kernel_t.at<float>(b + n) * (float)(input.at<C>(i + a, j + b)[2]);
							sum2_t_r += kernel_t.at<float>(b + n);
							sum2_t_g += kernel_t.at<float>(b + n);
							sum2_t_b += kernel_t.at<float>(b + n);
						}
					}
					sum_s_r += kernel_s.at<float>(a + n) * sum_t_r;
					sum_s_g += kernel_s.at<float>(a + n) * sum_t_g;
					sum_s_b += kernel_s.at<float>(a + n) * sum_t_b;

					sum2_s_r += kernel_s.at<float>(a + n) * sum2_t_r;
					sum2_s_g += kernel_s.at<float>(a + n) * sum2_t_g;
					sum2_s_b += kernel_s.at<float>(a + n) * sum2_t_b;


				}
				output.at<C>(i, j)[0] = (G)(sum_s_r / sum2_s_r);
				output.at<C>(i, j)[1] = (G)(sum_s_g / sum2_s_g);
				output.at<C>(i, j)[2] = (G)(sum_s_b / sum2_s_b);
			}
		}
	}
	return output;
}