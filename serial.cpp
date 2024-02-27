// minimizing libraries minimizes compile time
// this includes everything: #include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "omp.h"
#include <string>
#include <iostream>
#include <dirent.h>
#include <errno.h>

using namespace cv;
using namespace std;


// defines
#define errorexit(err_code, msg, line) do { cerr << msg << " " << line << endl; exit(err_code); } while (0)

// enums
enum mode {
	SERIAL = 0,
	PIPELINED = 1
};


// global vars
int retval;

string test = "test_images";
string results = "results";


Mat open_image(string filename)
{
	Mat image = imread(filename, IMREAD_GRAYSCALE);
	if (image.empty()) {
		cout << "Error: could not load image " << filename << endl;
		exit(1);
	}
	return image;
}

Mat gaussian_blur(Mat img)
{
	Mat img_blurred;
	double sigma = 1.0;		// must be in [1.0, 2.0]
	Size ksize = Size(6 * sigma + 1, 6 * sigma + 1);	// must be odd
	GaussianBlur(img, img_blurred, ksize, sigma);
	return img_blurred;
}

int belongs(double angle, double chosen_angle, double epsilon)
{
	double min = chosen_angle - epsilon;
	double max = chosen_angle + epsilon;
	// chosen_angle goes clockwise instead of counter clockwise
	if (min <= angle && angle < max)
		return 1;
	return 0;
}

Mat non_maximum_supression(Mat magnitude, Mat edge_direction)
{
	Mat img_suppressed = magnitude.clone();
	int h = magnitude.size().height;
	int w = magnitude.size().width;
	double *data = img_suppressed.ptr<double>();
	double *angle_data = edge_direction.ptr<double>();

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = y * w + x;
			double angle = angle_data[idx];
			double e = 22.5;
			double curr = data[idx];
			double n1 = 0;
			double n2 = 0;
			int n1x, n1y, n2x, n2y;

			if (belongs(angle, 0, e)) {
				n1x = x + 1;
				n1y = y;
				n2x = x - 1;
				n2y = y;
			} else if (belongs(angle, 45, e)) {
				n1x = x + 1;
				n1y = y + 1;
				n2x = x - 1;
				n2y = y - 1;
			} else if (belongs(angle, 90, e)) {
				n1x = x;
				n1y = y + 1;
				n2x = x;
				n2y = y - 1;
			} else if (belongs(angle, 135, e)) {
				n1x = x - 1;
				n1y = y + 1;
				n2x = x + 1;
				n2y = y - 1;
			} else if (belongs(angle, 180, e)) {
				n1x = x - 1;
				n1y = y;
				n2x = x + 1;
				n2y = y;
			} else if (belongs(angle, 225, e)) {
				n1x = x - 1;
				n1y = y - 1;
				n2x = x + 1;
				n2y = y + 1;
			} else if (belongs(angle, 270, e)) {
				n1x = x;
				n1y = y - 1;
				n2x = x;
				n2y = y + 1;
			} else if (belongs(angle, 315, e)) {
				n1x = x + 1;
				n1y = y - 1;
				n2x = x - 1;
				n2y = y + 1;
			} else if (belongs(angle, 360, e)) {
				n1x = x + 1;
				n1y = y;
				n2x = x - 1;
				n2y = y;
			} else {
				puts("This is not possible!");
			}

			
			// non-maximum suppression step
			// from current, neigh1, neigh2 choose max
			// suppress others
			if (0 <= n1x && n1x < w && 0 <= n1y && n1y < h)
				n1 = data[n1y * w + n1x];
			if (0 <= n2x && n2x < w && 0 <= n2y && n2y < h)
				n2 = data[n2y * w + n2x];
			if (curr < n1 || curr < n2)
				data[idx] = 0;
		}
	}
	return img_suppressed;
}

Mat threshold(Mat img, double low_ratio, double high_ratio)
{
	Mat img_thresholded = img.clone();
	int size = img_thresholded.cols * img_thresholded.rows;
	double *data = img_thresholded.ptr<double>();
	double high = 255 * high_ratio;
	double low = 255 * low_ratio;
	for (int i = 0; i < size; i++) {
		if (data[i] < low || high < data[i])
			data[i] = 0;
	}
	return img_thresholded;
}

Mat hysteresis(Mat img_weak, Mat img_strong)
{
	Mat img_hysteresed = img_weak.clone();
	double *weak = img_weak.ptr<double>();
	double *strong = img_strong.ptr<double>();
	double *hysteresed = img_hysteresed.ptr<double>();
	int size = img_weak.cols * img_weak.rows;
	int h = img_weak.size().height;
	int w = img_weak.size().width;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = y * w + x;
			if (strong[idx] > 0.000000001) {
				hysteresed[idx] = 255;
			}
			if (weak[idx] < 0.000000001) continue;
			// current pixel is weak, check all
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int nidx = (y + dy) * w + (x + dx);
					if (strong[nidx] > 0.000000001) {
						hysteresed[idx] = 255;
						// skip unnecessary checks
						goto cont;
					}
				}
			}
			hysteresed[idx] = 0;
cont:
			continue;
		}
	}
	return img_hysteresed;
}


/**
 * Serial proccessing of an image => 1 image at a time
 * TODO: make this function write in results dir, make a new subdir for every image and put all stage images in there
*/
void serial(string path) {

	double start, end;

	start = omp_get_wtime();
	Mat img = open_image(path);
	end = omp_get_wtime();
	printf("read time: %.3lf\n", end - start);

	// stage 1 - gaussian blur
	// input: grayscale image
	// output: blurred grayscale image
	start = omp_get_wtime();
	Mat img_blurred = gaussian_blur(img);
	end = omp_get_wtime();
	printf("stage1 time: %.3lf\n", end - start);

	// stage 2 - find gradient
	// input: blurred grayscale image
	// output: magnitude, edge_direction
	start = omp_get_wtime();
	Mat dx, dy;
	Sobel(img_blurred, dx, CV_64F, 1, 0, 3);
	Sobel(img_blurred, dy, CV_64F, 0, 1, 3);

	Mat magnitude, edge_direction;
	cartToPolar(dx, dy, magnitude, edge_direction, 1);
	end = omp_get_wtime();
	printf("stage2 time: %.3lf\n", end - start);

	// stage 3 - non-maximum suppression
	// input: magnitude, edge_direction
	// output: img_suppressed
	start = omp_get_wtime();
	Mat img_suppressed = non_maximum_supression(magnitude, edge_direction);
	end = omp_get_wtime();
	printf("stage3 time: %.3lf\n", end - start);

	// stage 4 - double threshold
	// input: img_suppressed
	// output: img_weak, img_strong
	start = omp_get_wtime();
	double high_ratio = 0.5;
	double low_ratio = 0.1;
	Mat img_strong = threshold(img_suppressed, high_ratio, 1);
	Mat img_weak = threshold(img_suppressed, low_ratio, high_ratio);
	end = omp_get_wtime();
	printf("stage4 time: %.3lf\n", end - start);

	// stage 5 - hysteresis
	// input: img_weak, img_strong
	// output: img_edges
	start = omp_get_wtime();
	Mat img_edges = hysteresis(img_weak, img_strong);
	end = omp_get_wtime();
	printf("stage5 time: %.3lf\n", end - start);

	start = omp_get_wtime();
	imwrite("results/processed.jpeg", img_edges);
	end = omp_get_wtime();
	printf("write time: %.3lf\n", end - start);
}

void get_files()
{
	string path;

	// test_images vars
	DIR* test_dir;
	struct dirent* entry;

	// results dir vars
	DIR* results_dir;

	// open test_images folder
	test_dir = opendir(test.c_str());
	if (test_dir == NULL) errorexit(errno, "opendir", __LINE__);

	// open results folder
	results_dir = opendir(results.c_str());
	if (results_dir == NULL) errorexit(errno, "opendir", __LINE__);

	// delete all results dir entries
	puts("Deleting all results dir entries...");
	while ((entry = readdir(results_dir)) != NULL) {
		// Skip . and ..
		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
			continue;
		}

		// Delete entry
		string delpath = results + (string) "/" + (string) entry->d_name;
		retval = remove(delpath.c_str());
		if (retval == -1) errorexit(errno, "remove", __LINE__);
	}

	while ((entry = readdir(test_dir)) != NULL) {
		// Skip . and ..
		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
			continue;
		}
		// Process image
		//printf("Queue: added image %s\n", entry->d_name);
		path = test + (string) "/" + (string) entry->d_name;
		serial(path);
	}

	// close dirs
	retval = closedir(test_dir);
	if (retval == -1) errorexit(errno, "closedir", __LINE__);
	retval = closedir(results_dir);
	if (retval == -1) errorexit(errno, "closedir", __LINE__);
}

/**
 * argv could contain enum indicating the mode (SERIAL, PIPELINED)
*/
int main(int argc, char** argv)
{
	double start, end;
	start = omp_get_wtime();
	get_files();
	end = omp_get_wtime();
	printf("full time: %.3lf\n", end - start);

	return 0;
}
