#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);
void hist_match(Mat &input, Mat &matched, G *trans_func_match, float *CDF, float *CDF_ref);

int main() {
	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("ref.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat ref_gray;
	cvtColor(input, input_gray, CV_RGB2GRAY);
	cvtColor(ref, ref_gray, CV_RGB2GRAY);

	Mat matched = input_gray.clone(); // result matrix

	FILE *f_PDF;
	FILE *f_dPDF;
	FILE *f_matched_PDF_gray;
	FILE *f_trans_func_match;

	
	G trans_func_match[L] = { 0 }; // transfer function of matched image

	fopen_s(&f_PDF, "PDF2.txt", "w+");
	fopen_s(&f_dPDF, "desired_PDF.txt", "w+");
	fopen_s(&f_matched_PDF_gray, "hist_matched_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_match, "trans_func_match.txt", "w+");

	float *PDF = cal_PDF(input_gray);
	float *CDF = cal_CDF(input_gray);

	float *PDF_ref = cal_PDF(ref_gray);
	float *CDF_ref = cal_CDF(ref_gray);
	
	hist_match(input, matched, trans_func_match, CDF, CDF_ref); // histogram matching

	float *matched_PDF_gray = cal_PDF(matched);
	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_dPDF, "%d\t%f\n", i, PDF_ref[i]);
		fprintf(f_matched_PDF_gray, "%d\t%f\n", i, matched_PDF_gray[i]);

		// write transfer functions
		fprintf(f_trans_func_match, "%d\t%d\n", i, trans_func_match[i]);
	}

	// memory release
	free(PDF);
	free(CDF);
	free(PDF_ref);
	free(CDF_ref);
	fclose(f_PDF);
	fclose(f_dPDF);
	fclose(f_matched_PDF_gray);
	fclose(f_trans_func_match);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Reference", WINDOW_AUTOSIZE);
	imshow("Reference", ref_gray);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);

	waitKey(0);

	return 0;

}
void hist_match(Mat &input, Mat &matched, G *trans_func_match, float *CDF, float *CDF_ref) {

	G trans_func_ref[L] = { 0 }; // transfer function of referenced image
	G trans_func[L] = { 0 };  // transfer function of original image

	// compute transfer function of referenced image
	for (int i = 0; i < L; i++){
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
				lookup[i] = idx-1;
			else lookup[i] = idx;
		}
	}

	for (int i = 0; i < L; i++) { // compute transfer function of matched image
		trans_func_match[i] = trans_func_ref[lookup[i]];
	}
	
	
	for (int i = 0; i < input.rows; i++) {
		for(int j=0; j< input.cols; j++) // mapping values into a result matrix
			matched.at<G>(i, j) = trans_func_match[input.at<G>(i, j)];
	}
}

