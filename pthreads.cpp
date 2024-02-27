#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "omp.h"
#include <string>
#include <iostream>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <queue>

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
					if (nidx < 0 || nidx > size) continue;
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

///////////////////////////////////////////////////////////////////////////////
// begining of PTHREAD code
///////////////////////////////////////////////////////////////////////////////

string test = "test_images";
string results = "results";

int num_pictures_to_process = 0;
int num_processed = 0;

int todo_read = 0;
int todo_stage_1 = 0;
int todo_stage_2 = 0;
int todo_stage_3 = 0;
int todo_stage_4 = 0;
int todo_stage_5 = 0;
int todo_write = 0;

queue<string> filenames;
queue<Mat> q01;	// read from file
queue<Mat> q12;
queue<Mat> q23;
queue<Mat> q34;
queue<Mat> q45;
queue<Mat> q56;	// write to file

pthread_mutex_t filenames_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q01_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q12_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q23_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q34_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q45_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t q56_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t generic_mutex = PTHREAD_MUTEX_INITIALIZER;

// this one is special (polling isn't necessary)
void *file_opener_thread(void *arg)
{
	while (todo_read > 0) {
		string filename;
		Mat img;

		pthread_mutex_lock(&filenames_mutex);
		// ran out of work
		if (filenames.empty()) {
			pthread_mutex_unlock(&filenames_mutex);
			return NULL;
		}

		filename = filenames.front();
		filenames.pop();
		todo_read--;
		pthread_mutex_unlock(&filenames_mutex);

		// biggest operation
		img = open_image(filename);

		pthread_mutex_lock(&q01_mutex);
		q01.push(img);
		pthread_mutex_unlock(&q01_mutex);
	}
	return NULL;
}

void *gaussian_blur_thread(void *arg)
{
	// stage 1 - gaussian blur
	// input: grayscale image
	// output: blurred grayscale image
	
	while (todo_stage_1 > 0) {
		Mat img, img_blurred;

		// polling for next
		while (todo_stage_1 > 0) {
			pthread_mutex_lock(&q01_mutex);
			if (q01.size() > 0) break;
			pthread_mutex_unlock(&q01_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_stage_1 <= 0) return NULL;

		// next is available (while breaked), mutex locked
		img = q01.front();
		q01.pop();
		todo_stage_1--;
		pthread_mutex_unlock(&q01_mutex);
		
		// biggest operation
		img_blurred = gaussian_blur(img);

		pthread_mutex_lock(&q12_mutex);
		q12.push(img_blurred);
		pthread_mutex_unlock(&q12_mutex);
	}
	return NULL;
}

void *find_gradient_thread(void *arg)
{
	// stage 2 - find gradient
	// input: blurred grayscale image
	// output: magnitude, edge_direction

	while (todo_stage_2 > 0) {
		Mat img_blurred, magnitude, edge_direction;
		Mat dx, dy;

		// polling for next
		while (todo_stage_2 > 0) {
			pthread_mutex_lock(&q12_mutex);
			if (q12.size() > 0) break;
			pthread_mutex_unlock(&q12_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_stage_2 <= 0) return NULL;

		// next is available (while breaked), mutex locked
		img_blurred = q12.front();
		q12.pop();
		todo_stage_2--;
		pthread_mutex_unlock(&q12_mutex);
		
		// biggest operation
		Sobel(img_blurred, dx, CV_64F, 1, 0, 3);
		Sobel(img_blurred, dy, CV_64F, 0, 1, 3);
		cartToPolar(dx, dy, magnitude, edge_direction, 1);

		pthread_mutex_lock(&q23_mutex);
		q23.push(magnitude);
		q23.push(edge_direction);
		pthread_mutex_unlock(&q23_mutex);
	}
	return NULL;
}

void *non_max_suppression_thread(void *arg)
{
	// stage 3 - non-maximum suppression
	// input: magnitude, edge_direction
	// output: img_suppressed

	while (todo_stage_3 > 0) {
		Mat magnitude, edge_direction, img_suppressed;

		// polling for next
		while (todo_stage_3 > 0) {
			pthread_mutex_lock(&q23_mutex);
			if (q23.size() > 0) break;
			pthread_mutex_unlock(&q23_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_stage_3 <= 0) return NULL;

		// next is available (while breaked), mutex locked
		magnitude = q23.front();
		q23.pop();
		edge_direction = q23.front();
		q23.pop();
		todo_stage_3--;
		pthread_mutex_unlock(&q23_mutex);
		
		// biggest operation
		img_suppressed = non_maximum_supression(magnitude, edge_direction);

		pthread_mutex_lock(&q34_mutex);
		q34.push(img_suppressed);
		pthread_mutex_unlock(&q34_mutex);
	}
	return NULL;
}

void *threshold_thread(void *arg)
{
	// stage 4 - double threshold
	// input: img_suppressed
	// output: img_weak, img_strong
	double high_ratio = 0.16;
	double low_ratio = 0.08;

	while (todo_stage_4 > 0) {
		Mat img_suppressed, img_weak, img_strong;


		// polling for next
		while (todo_stage_4 > 0) {
			pthread_mutex_lock(&q34_mutex);
			if (q34.size() > 0) break;
			pthread_mutex_unlock(&q34_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_stage_4 <= 0) return NULL;

		// next is available (while breaked), mutex locked
		img_suppressed = q34.front();
		q34.pop();
		todo_stage_4--;
		pthread_mutex_unlock(&q34_mutex);
		
		// biggest operation
		img_strong = threshold(img_suppressed, high_ratio, 1);
		img_weak = threshold(img_suppressed, low_ratio, high_ratio);

		pthread_mutex_lock(&q45_mutex);
		q45.push(img_weak);
		q45.push(img_strong);
		pthread_mutex_unlock(&q45_mutex);
	}
	return NULL;
}

void *hysteresis_thread(void *arg)
{
	// stage 5 - hysteresis
	// input: img_weak, img_strong
	// output: img_edges

	while (todo_stage_5 > 0) {
		Mat img_weak, img_strong, img_edges;

		// polling for next
		while (todo_stage_5 > 0) {
			pthread_mutex_lock(&q45_mutex);
			if (q45.size() > 0) break;
			pthread_mutex_unlock(&q45_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_stage_5 <= 0) return NULL;

		// next is available (while breaked), mutex locked
		img_weak = q45.front();
		q45.pop();
		img_strong = q45.front();
		q45.pop();
		todo_stage_5--;
		pthread_mutex_unlock(&q45_mutex);
		
		// biggest operation
		img_edges = hysteresis(img_weak, img_strong);

		pthread_mutex_lock(&q56_mutex);
		q56.push(img_edges);
		pthread_mutex_unlock(&q56_mutex);
	}

	return NULL;
}

void *write_file_thread(void *arg)
{
	while (todo_write > 0) {
		Mat img_edges;

		int current_processed;
		char name[100];

		// polling for next
		while (todo_write > 0) {
			pthread_mutex_lock(&q56_mutex);
			if (q56.size() > 0) break;
			pthread_mutex_unlock(&q56_mutex);
		}
		// finished (while exited), mutex already unlocked
		if (todo_write <= 0) return NULL;

		// next is available (while breaked), mutex locked
		img_edges = q56.front();
		q56.pop();
		todo_write--;
		pthread_mutex_unlock(&q56_mutex);

		pthread_mutex_lock(&generic_mutex);
		current_processed = num_processed;
		num_processed++;
		pthread_mutex_unlock(&generic_mutex);
		
		sprintf(name, "%s/processed%d.jpeg", results.c_str(), current_processed);

		// biggest operation
		imwrite(name, img_edges);
	}
	return NULL;
}

// nthreads is array that tells how many threads per stage
void pthreads(int *nthreads)
{
	int stage = 0;

	pthread_t file_opener[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&file_opener[i], NULL, file_opener_thread, NULL);
	}
	stage++;

	pthread_t stage_1[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&stage_1[i], NULL, gaussian_blur_thread, NULL);
	}
	stage++;

	pthread_t stage_2[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&stage_2[i], NULL, find_gradient_thread, NULL);
	}
	stage++;

	pthread_t stage_3[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&stage_3[i], NULL, non_max_suppression_thread, NULL);
	}
	stage++;

	pthread_t stage_4[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&stage_4[i], NULL, threshold_thread, NULL);
	}
	stage++;

	pthread_t stage_5[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&stage_5[i], NULL, hysteresis_thread, NULL);
	}
	stage++;


	pthread_t file_writer[nthreads[stage]];
	for (int i = 0; i < nthreads[stage]; i++) {
		pthread_create(&file_writer[i], NULL, write_file_thread, NULL);
	}
	stage++;

	// waiting for threads
	for (int i = 0; i < nthreads[0]; i++) {
		pthread_join(file_opener[i], NULL);
		//cout << "waited0" << endl;
	}
	for (int i = 0; i < nthreads[1]; i++) {
		pthread_join(stage_1[i], NULL);
		//cout << "waited1" << endl;
	}
	for (int i = 0; i < nthreads[2]; i++) {
		pthread_join(stage_2[i], NULL);
		//cout << "waited2" << endl;
	}
	for (int i = 0; i < nthreads[3]; i++) {
		pthread_join(stage_3[i], NULL);
		//cout << "waited3" << endl;
	}
	for (int i = 0; i < nthreads[4]; i++) {
		pthread_join(stage_4[i], NULL);
		//cout << "waited4" << endl;
	}
	for (int i = 0; i < nthreads[5]; i++) {
		pthread_join(stage_5[i], NULL);
		//cout << "waited5" << endl;
	}
	for (int i = 0; i < nthreads[6]; i++) {
		pthread_join(file_writer[i], NULL);
		//cout << "waited6" << endl;
	}
	
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
	//puts("Deleting all results dir entries...");
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
		filenames.push(path);
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
	int nthreads[7];
	// set up how many threads per stage
	nthreads[0] = 2;	// reading
	nthreads[1] = 1;	// gaussian blur
	nthreads[2] = 2;	// find gradient
	nthreads[3] = 6;	// non-maximum supression
	nthreads[4] = 2;	// thresholding
	nthreads[5] = 1;	// hysteresis
	nthreads[6] = 1;	// writing
	// 1123111
	// 2126211 -> 0.15 * nthreads je prblizn ns cas

	get_files();
	double start = omp_get_wtime();
	num_pictures_to_process = filenames.size();
	todo_read = num_pictures_to_process;
	todo_stage_1 = num_pictures_to_process;
	todo_stage_2 = num_pictures_to_process;
	todo_stage_3 = num_pictures_to_process;
	todo_stage_4 = num_pictures_to_process;
	todo_stage_5 = num_pictures_to_process;
	todo_write = num_pictures_to_process;

	pthreads(nthreads);

	double end = omp_get_wtime();
	printf("time: %.3lf\n", end - start);
	return 0;
}

// opening files is special, because queue already complete

// queue has elements


// others have to poll the queue

// queue is empty
// 	polling for next element
// 	finished
// queue has elements
