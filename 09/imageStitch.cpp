#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.5

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
Mat cal_affine(vector<Point2f>& ptl, vector<Point2f>& ptr, int number_of_points);
void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);
int main() {

	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));
	

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));
	
	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		kp.pt.x += size.width;
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints; // left image
	vector<Point2f> dstPoints; // right image
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());
	
	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point2f(size.width + pt2.x, pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255));
	}
	
	// Display mathing image
	namedWindow("Matching1");
	imshow("Matching1", matchingImage);



	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);

	Mat A12 = cal_affine(dstPoints, srcPoints, dstPoints.size());
	Mat A21 = cal_affine(srcPoints, dstPoints, srcPoints.size());
	
	std::cout << A12 << std::endl;
	std::cout << A21 << std::endl;
	// height(row), width(col) of each image
	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;

	// compute corners (p1, p2, p3, p4)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_col + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * I2_col + A21.at<float>(5));

	Point2f p3(A21.at<float>(0) * I2_row + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * I2_row + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p4(A21.at<float>(0) * I2_row + A21.at<float>(1) * I2_col + A21.at<float>(2),
		A21.at<float>(3) * I2_row + A21.at<float>(4) * I2_col + A21.at<float>(5));

	// for inverse warping
	Point2f p1_(A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p2_(A12.at<float>(0) * 0 + A12.at<float>(1) * I1_col + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * I1_col + A12.at<float>(5));

	Point2f p3_(A12.at<float>(0) * I1_row + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * I1_row + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p4_(A12.at<float>(0) * I1_row + A12.at<float>(1) * I1_col + A12.at<float>(2),
		A12.at<float>(3) * I1_row + A12.at<float>(4) * I1_col + A12.at<float>(5));

	std::cout << p1 << p2 << p3 << p4 << std::endl;
	std::cout << p1_ << p2_ << p3_ << p4_ << std::endl;
	
	// compute boundary for merged image(I_f)
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(std::max(I1_row, std::max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(std::max(I1_col, std::max(p2.y, p4.y)));
	std::cout << bound_u << " " << bound_b << " " << bound_l << " " << bound_r << std::endl;

	// compute boundary for inverse warping
	int bound_u_ = (int)round(min(0.0f, min(p1_.x, p2_.x)));
	int bound_b_ = (int)round(std::max(I2_row, std::max(p3_.x, p4_.x)));
	int bound_l_ = (int)round(min(0.0f, min(p1_.y, p3_.y)));
	int bound_r_ = (int)round(std::max(I2_col, std::max(p2_.y, p4_.y)));
	std::cout << bound_u_ << " " << bound_b_ << " " << bound_l_ << " " << bound_r_ << std::endl;

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);

	int diff_x_ = abs(bound_u_);
	int diff_y_ = abs(bound_l_);

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));
	std::cout << I_f.size() << std::endl;
	
	// inverse warping with bilinear interplolation
	for (int i = -diff_x_; i < I_f.rows - diff_x_; i++) {
		for (int j = -diff_y_; j < I_f.cols - diff_y_; j++) {
			float x = A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2) + diff_x_;
			float y = A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5) + diff_y_;
			// get nearest two points
			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			// calculate the ratio of the regions
			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_row && y1 >= 0 && y2 < I2_col) { // only valid points
				// intermediate coordinates for bilinear interpolation

				Vec3f p = mu * input2.at<Vec3f>(x1, y2) + (1 - mu)*input2.at<Vec3f>(x1, y1);
				Vec3f q = mu * input2.at<Vec3f>(x2, y2) + (1 - mu) *input2.at<Vec3f>(x2, y1);
				

				I_f.at<Vec3f>(i + diff_x_, j + diff_y_) = lambda * q + (1 - lambda) * p;
				

			}
		}
	}

	// image stitching with blend
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);


	namedWindow("result");
	imshow("result", I_f);

	
	waitKey(0);

	return 0;
}

/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;

	for (int i = 0; i < dim; i++) {
		//std::cout << vec1.at<uchar>(i,0) << std::endl;
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		double d = euclidDistance(vec, v);

		if (d < minDist) {
			minDist = d;
			neighbor = i;
		}

	}


	return neighbor;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {
			double minDist = euclidDistance(desc1, descriptors2.row(0));
			double SecminDist = euclidDistance(desc1, descriptors2.row(1));
			double temp = 0;
			if (minDist > SecminDist) {
				temp = minDist;
				minDist = SecminDist;
				SecminDist = temp;
			}
			for (int j = 2; j < descriptors2.rows; j++) {
				Mat v = descriptors2.row(j);		// each row of descriptor
				double d = euclidDistance(desc1, v);
				//std::cout << "D: " << d << " minDist : " << minDist << " SecminDist : " << SecminDist<< std::endl;
				if (d < minDist) {
					SecminDist = minDist;
					minDist = d;
				} 
				else if (d < SecminDist && d != minDist) {
					SecminDist = d;
				}
			}
			double dist1 = minDist;
			double dist2 = SecminDist;
			

			if (dist1 / dist2 >= RATIO_THR)
				continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn);
			int nn2 = nearestNeighbor(desc2, keypoints1, descriptors1);

			if (i != nn2)
				continue;
		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
}
Mat cal_affine(vector<Point2f>& ptl, vector<Point2f>& ptr, int number_of_points) {
	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {
		/*
		M = (y, x, 1, 0, 0, 0)
			(0, 0, 0, y, x, 1)
		*/

		M.at<float>(2 * i, 0) = ptl[i].y;
		M.at<float>(2 * i, 1) = ptl[i].x;
		M.at<float>(2 * i, 2) = 1.0;
		M.at<float>(2 * i + 1, 3) = ptl[i].y;
		M.at<float>(2 * i + 1, 4) = ptl[i].x;
		M.at<float>(2 * i + 1, 5) = 1.0;

		b.at<float>(2 * i) = ptr[i].y;
		b.at<float>(2 * i + 1) = ptr[i].x;

	}
	
	// affine x = (M.T*M)(-1)*M.T*b
	M_trans = M.t();
	temp = (M_trans*M).inv();
	affineM = temp * M_trans*b;
	return affineM;
}
void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;
			// I2 is already in I_f by inverse warping
			// So, It is not necessary to check that only I2 is valid
			// if both are valid
			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);
			}
			// only I1 is valid
			else if (cond1) {
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i - diff_x, j - diff_y);
			}
		}
	}
}