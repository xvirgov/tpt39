#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define STRING_BUFFER_LEN 1024

using namespace std;

void print_matrix(float *matrix, int m, int n) {
	printf("------------------------------\n");
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			printf("%f, ", matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("------------------------------\n");
}

void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

long timespec_ms(struct timespec *start, struct timespec *end) {

	long nsms = (end->tv_nsec - start->tv_nsec)/ (1000 * 1000);
	if (nsms < 0)
		nsms += 1;

	return (end->tv_sec - start->tv_sec) * 1000 + nsms ;
}

int main()
{
	int errcode = CL_SUCCESS;
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     {
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------
const unsigned N = 3;
const unsigned M = 3;
float *input_a=(float *) malloc(sizeof(float)*N*M);
float *input_b=(float *) malloc(sizeof(float)*N*M);
float *output=(float *) malloc(sizeof(float)*N*M);
float *ref_output=(float *) malloc(sizeof(float)*N*M);
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;

struct timespec tstart, tend;
struct timespec tstart_cpy, tend_cpy;
struct timespec tstart_kern, tend_kern;
// struct timespec tstart_write, tend_write;
long ms = 0, msall = 0, ms_cpy = 0, ms_kern = 0;

clGetPlatformIDs(1, &platform, NULL);
clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

context_properties[1] = (cl_context_properties)platform;
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
queue = clCreateCommandQueue(context, device, 0, NULL);

unsigned char **opencl_program=read_file("matrix_mult.cl");
program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
if (program == NULL){
	printf("Program creation failed\n");
  return 1;
}

	if(clock_gettime(CLOCK_MONOTONIC_RAW, &tstart) < 0) {
		return -1;
	}

	if(clock_gettime(CLOCK_MONOTONIC_RAW, &tstart_cpy) < 0) {
		return -1;
	}

     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_add", NULL);

 // Input buffers.
 input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
			N*M* sizeof(float), NULL, &status);
 checkError(status, "Failed to create buffer for input A");

 input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
		 N*M* sizeof(float), NULL, &status);
 checkError(status, "Failed to create buffer for input B");

	 // Output buffer.
	 output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
			 N*M* sizeof(float), NULL, &status);
	 checkError(status, "Failed to create buffer for output");

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
		cl_event kernel_event,finish_event;

		input_a = (float *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE,
				CL_MAP_WRITE,0, N* sizeof(float), 0, NULL, &write_event[0],&errcode);
		checkError(errcode, "Failed to map input A");

		input_b = (float *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE,
				CL_MAP_WRITE, 0,N* sizeof(float), 0, NULL, &write_event[1],&errcode);
		checkError(errcode, "Failed to map input B");

		// Map to host memory
		output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,
				CL_MAP_READ, 0,N*M* sizeof(float),  0, NULL, NULL,&errcode);
		checkError(errcode, "Failed to map output");

		// for(unsigned j = 0; j < N*M; ++j) {
		// 			input_a[j] = 2; //rand_float();
		// 			input_b[j] = 2; //rand_float();
		// }
		input_a[0] = 1;
		input_a[1] = 2;
		input_a[2] = 3;
		input_a[3] = 4;
		input_a[4] = 5;
		input_a[5] = 6;
		input_a[6] = 7;
		input_a[7] = 8;
		input_a[8] = 9;

		input_b[0] = 10;
		input_b[1] = 11;
		input_b[2] = 12;
		input_b[3] = 13;
		input_b[4] = 14;
		input_b[5] = 15;
		input_b[6] = 16;
		input_b[7] = 17;
		input_b[8] = 18;
		print_matrix(input_a, M, N);

		print_matrix(input_b, M, N);

			if(clock_gettime(CLOCK_MONOTONIC_RAW, &tstart) < 0) {
				return -1;
			}

			for(unsigned j = 0; j < N*M; ++j) {
						ref_output[j] = input_a[j] + input_b[j];
						// printf("ref %f\n",ref_output[j]);
			}

			if(clock_gettime(CLOCK_MONOTONIC_RAW, &tend) < 0) {
				return -1;
			}

					 ms = timespec_ms(&tstart, &tend);
			// time (&end);
			// diff = difftime (end,start);
				printf ("CPU took %ld miliseconds to run.\n", ms );



    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

		status = clSetKernelArg(kernel, argi++, sizeof(unsigned), &N);
    checkError(status, "Failed to set argument 4");

		status = clSetKernelArg(kernel, argi++, sizeof(unsigned), &M);
    checkError(status, "Failed to set argument 4");

		clWaitForEvents(2, write_event);
		if(clock_gettime(CLOCK_MONOTONIC_RAW, &tend_cpy) < 0) {
			return -1;
		}

		if(clock_gettime(CLOCK_MONOTONIC_RAW, &tstart_kern) < 0) {
			return -1;
		}

	clEnqueueUnmapMemObject(queue,input_a_buf,input_a,0,NULL,NULL);
	clEnqueueUnmapMemObject(queue,input_b_buf,input_b,0,NULL,NULL);

	const size_t global_work_size[2] = {M,N};
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
      global_work_size, NULL, 2, write_event, &kernel_event);

  checkError(status, "Failed to launch kernel");
	status=clWaitForEvents(1,&kernel_event);
	checkError(status, "Failed  wait");

	if(clock_gettime(CLOCK_MONOTONIC_RAW, &tend_kern) < 0) {
		return -1;
	}

	//  if(clock_gettime(CLOCK_MONOTONIC_RAW, &tstart_write) < 0) {
	//  	return -1;
	//  }
    // Read the result. This the final operation.
    // status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
    //     0, N* sizeof(float), output, 1, &kernel_event, &finish_event);

clWaitForEvents(1, &finish_event);

// SIGSEGV
// if(clock_gettime(CLOCK_MONOTONIC_RAW, &tend_write) < 0) {
// 	return -1;
// }

if(clock_gettime(CLOCK_MONOTONIC_RAW, &tend) < 0) {
	return -1;
}

msall = timespec_ms(&tstart, &tend);
ms_cpy = timespec_ms(&tstart_cpy, &tend_cpy);
ms_kern = timespec_ms(&tstart_kern, &tend_kern);
// ms_write = timespec_ms(&tstart_kern, &tend_write);

printf ("GPU took %ld miliseconds to run. (ALL::[%ld], COPY::[%ld], WRITE::[])\n", ms_kern, msall, ms_cpy);
// Verify results.
bool pass = true;

for(unsigned j = 0; j < N*M && pass; ++j) {
      if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
            j, output[j], ref_output[j]);
        pass = false;
      }
}
    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(input_b_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);


//--------------------------------------------------------------------






     clFinish(queue);

     return 0;
}
