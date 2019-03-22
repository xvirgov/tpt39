#ifndef GPU_FILTERS_HPP
#define GPU_FILTERS_HPP

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// helper funcs
void checkError(int status, const char *msg);
void print_clbuild_errors(cl_program program,cl_device_id device);

// filters
void gaussianBlur_gpu(Mat frame, Mat result, Mat gaussKernel);

#endif  // GPU_FILTERS_HPP
