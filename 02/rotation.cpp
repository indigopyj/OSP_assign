#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;


	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);
	destroyAllWindows();
	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180; // degree를 radian으로 변환

	float sq_row = ceil(row * sin(radian) + col * cos(radian)); // enlargement of row
	float sq_col = ceil(col * sin(radian) + row * cos(radian)); // enlargment of col

	Mat output = Mat::zeros(sq_row, sq_col, input.type()); // matrix for output

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			// 좌표 회전
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;
			// inverse warping
			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) { // Nearest neighbor interpolation
					x = round(x);
					y = round(y);
					output.at<Vec3b>(i, j) = input.at<Vec3b>(y, x);


				}
				else if (!strcmp(opt, "bilinear")) { // Bilinear interpolation
					// get nearest two points
					float y1 = floor(y);
					float y2 = ceil(y);
					float x1 = floor(x);
					float x2 = ceil(x);

					// calculate the ratio of the regions
					float mu = y - y1;
					float lambda = x - x1;

					// intermediate coordinates for bilinear interpolation
					float py = mu * y2 + (1 - mu)*y1;
					float px = mu * x1 + (1 - mu)*x1;
					float qy = mu * y2 + (1 - mu)*y1;
					float qx = mu * x2 + (1 - mu)*x2;
					// interpolate the points
					float x = lambda * qx + (1 - lambda)*px;
					float y = lambda * qy + (1 - lambda)*py;
					// mapping into output matrix
					output.at<Vec3b>(i, j) = input.at<Vec3b>(y, x);

				}

			}
		}
	}

	return output;
}