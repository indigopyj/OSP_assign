#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
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

// generate PDF for single channel image
float *cal_PDF(Mat &input) {

	int count[L] = { 0 };
	float *PDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute PDF
	for (int i = 0; i < L; i++)
		PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

	return PDF;
}

// generate PDF for color image
float **cal_PDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **PDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		PDF[i] = (float*)calloc(3, sizeof(float)); // float 3°³ ÇÒ´ç.

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////
	
	
	// Count
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			count[input.at<C>(i, j)[0]][0]++;
			count[input.at<C>(i, j)[1]][1]++;
			count[input.at<C>(i, j)[2]][2]++;
		}
	}

	// Compute PDF
	for (int i = 0; i < L; i++) {
		PDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols);
		PDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols);
		PDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols);
	}

	return PDF;
}

// generate CDF for single channel image
float *cal_CDF(Mat &input) {

	int count[L] = { 0 };
	float *CDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute CDF
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)// cumulatively summation
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}

// generate CDF for color image
float **cal_CDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	// Count
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			count[input.at<C>(i, j)[0]][0]++; // channel B
			count[input.at<C>(i, j)[1]][1]++; // channel G
			count[input.at<C>(i, j)[2]][2]++; // channel R
		}
	}

	// Compute CDF
	for (int i = 0; i < L; i++) {
		CDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols); // channel B
		CDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols); // channel G
		CDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols); // channel R

		if (i != 0) { // cumulatively summation
			CDF[i][0] += CDF[i - 1][0];
			CDF[i][1] += CDF[i - 1][1];
			CDF[i][2] += CDF[i - 1][2];
		}
	}

	return CDF;
}