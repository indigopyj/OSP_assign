#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);
void hist_match(Mat &input, Mat &matched, G *trans_func_match, float *CDF, float *CDF_ref);

int main() {
	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("ref2.jpg", CV_LOAD_IMAGE_COLOR);

	Mat ref_YUV;
	Mat matched_YUV; // result matrix

	// RGB -> YUV
	cvtColor(input, matched_YUV, CV_RGB2YUV);
	cvtColor(ref, ref_YUV, CV_RGB2YUV);
	// split each channel(Y, U, V)
	Mat channels[3], channels_ref[3];
	split(matched_YUV, channels); split(ref_YUV, channels_ref);
	Mat Y = channels[0];
	Mat Y_ref = channels_ref[0];
	


	// PDF or transfer function txt files
	FILE *f_equalized_PDF_YUV, *f_PDF_RGB, *f_PDF_RGB_ref;
	FILE *f_trans_func_match_YUV;

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image

	float **PDF_RGB_ref = cal_PDF_RGB(ref);		// PDF of Reference image(RGB) : [L][3]
	float *CDF_YUV_ref = cal_CDF(Y_ref);				// CDF of Y channel image

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_PDF_RGB_ref, "PDF_RGB_ref.txt", "w+");
	fopen_s(&f_equalized_PDF_YUV, "hist_matched_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_match_YUV, "trans_func_match_YUV.txt", "w+");

	G trans_func_match_YUV[L] = { 0 };			// transfer function of matched image


	hist_match(Y, Y, trans_func_match_YUV, CDF_YUV, CDF_YUV_ref); // histogram matching

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);
	merge(channels_ref, 3, ref_YUV);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);
	cvtColor(ref_YUV, ref_YUV, CV_YUV2RGB);


	float **matched_PDF_YUV = cal_PDF_RGB(matched_YUV); // calculate PDF of matched image
	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_PDF_RGB_ref, "%d\t%f\t%f\t%f\n", i, PDF_RGB_ref[i][0], PDF_RGB_ref[i][1], PDF_RGB_ref[i][2]);
		fprintf(f_equalized_PDF_YUV, "%d\t%f\t%f\t%f\n", i, matched_PDF_YUV[i][0], matched_PDF_YUV[i][1], matched_PDF_YUV[i][2]);

		// write transfer functions
		fprintf(f_trans_func_match_YUV, "%d\t%d\n", i, trans_func_match_YUV[i]);
	}

	// memory release
	free(PDF_RGB);
	free(PDF_RGB_ref);
	free(CDF_YUV);
	free(CDF_YUV_ref);
	fclose(f_PDF_RGB);
	fclose(f_PDF_RGB_ref);
	fclose(f_equalized_PDF_YUV);
	fclose(f_trans_func_match_YUV);

	namedWindow("Color", WINDOW_AUTOSIZE);
	imshow("Color", input);

	namedWindow("Reference", WINDOW_AUTOSIZE);
	imshow("Reference", ref);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched_YUV);

	waitKey(0);

	return 0;

}
void hist_match(Mat &input, Mat &matched, G *trans_func_match, float *CDF, float *CDF_ref) {

	G trans_func_ref[L] = { 0 }; // transfer function of referenced image
	G trans_func[L] = { 0 };  // transfer function of original image

	// compute transfer function of referenced image
	for (int i = 0; i < L; i++) {
		trans_func[i] = (G)((L - 1) * CDF[i]);
	}

	// compute transfer function of original image
	for (int i = 0; i < L; i++) {
		trans_func_ref[i] = (G)((L - 1) * CDF_ref[i]);
	}


	// find nearest values and make lookup table to find mapped value to referenced histogram
	int idx = 0;
	int lookup[L] = { 0 };
	for (int i = 0; i < L; i++) {
		if (trans_func[i] <= trans_func_ref[idx]) lookup[i] = idx;
		else {
			while (trans_func[i] > trans_func_ref[idx]) idx++;
			if (trans_func_ref[idx] - trans_func[i] > trans_func[i] - trans_func_ref[idx - 1])
				// if previous one is much nearer,
				lookup[i] = idx - 1;
			else lookup[i] = idx;
		}
	}

	for (int i = 0; i < L; i++) { // compute transfer function of matched image
		trans_func_match[i] = trans_func_ref[lookup[i]];
	}


	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) // mapping values into a result matrix
			matched.at<G>(i, j) = trans_func_match[input.at<G>(i, j)];
	}
}

