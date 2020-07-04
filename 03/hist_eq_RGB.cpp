#include "hist_func.h"

void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat equalized_RGB = input.clone();

	// PDF or transfer function txt files
	FILE *f_equalized_PDF_RGB, *f_PDF_RGB;
	FILE *f_trans_func_eq_RGB;

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");

	float **PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float **CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function

	// histogram equalization on RGB image
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	// equalized PDF (RGB)
	float **equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_RGB, "%d\t%f\t%f\t%f\n", i, equalized_PDF_RGB[i][0], equalized_PDF_RGB[i][1], equalized_PDF_RGB[i][2]);

		// write transfer functions
		fprintf(f_trans_func_eq_RGB, "%d\t%d\t%d\t%d\n", i, trans_func_eq_RGB[i][0], trans_func_eq_RGB[i][1], trans_func_eq_RGB[i][2]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) {

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	// compute transfer function
	for (int i = 0; i < L; i++) {
		trans_func[i][0] = (G)((L - 1) * CDF[i][0]); // channel B
		trans_func[i][1] = (G)((L - 1) * CDF[i][1]); // channel G
		trans_func[i][2] = (G)((L - 1) * CDF[i][2]); // channel R
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			equalized.at<C>(i, j)[0] = trans_func[input.at<C>(i, j)[0]][0]; // channel B
			equalized.at<C>(i, j)[1] = trans_func[input.at<C>(i, j)[1]][1]; // channel G
			equalized.at<C>(i, j)[2] = trans_func[input.at<C>(i, j)[2]][2]; // channel R

		}
	}

}