#include <stdio.h>
#include <OpenCL/OpenCL.h>
#include <stdlib.h>
#include <time.h>

const char* vecAdd =
"__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result, int len){int idx = get_global_id(0);if(idx < len){result[idx] = a[idx] + b[idx];}}";

#define cl_check(err)                                        \
do{                                                          \
if (err != CL_SUCCESS){                                      \
printf("ERROR code %d on line %d\n", err, __LINE__ -1);      \
return -1;                                                   \
}                                                            \
} while (0)

int main(int argc, char const *argv[])
{
    int vectorLength = 40960000;
    float* host_a = (float*) malloc(sizeof(float) * vectorLength);
    float* host_b = (float*) malloc(sizeof(float) * vectorLength);
    float* host_c = (float*) malloc(sizeof(float) * vectorLength);
    float* host_validation = (float*) malloc(sizeof(float) * vectorLength);
    
    for(int i = 0; i < vectorLength; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }
    
    clock_t start = clock(), diff;
    for(int i = 0; i < vectorLength; i++)
    {
        host_validation[i] = host_a[i] + host_b[i];
    }
    diff = clock() - start;
    printf("Computation takes %lu milliseconds. \n", diff * 1000 / CLOCKS_PER_SEC);
    
    
    start = clock();
    cl_int status = -1;
    //cl_int type to receive the error code
    
    cl_context context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    //create a openCL context contains all devices
    cl_check(status);
    
    size_t infoSize;
    status = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &infoSize); //size of devices information
    cl_check(status);
    
    cl_device_id* devices = (cl_device_id*) malloc(infoSize);
    status = clGetContextInfo(context, CL_CONTEXT_DEVICES, infoSize, devices, NULL);
    cl_check(status);
    //write device index into the array
    
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &status);
    cl_check(status);
    // create a command queue for the first device
    
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &vecAdd, NULL, &status);
    cl_check(status);
    //OpenCL kernel code as a big string
    
    char compileFlags[4096];
    sprintf(compileFlags, "-cl-mad-enable");
    status = clBuildProgram(program, 0, NULL, compileFlags, NULL, NULL);
    cl_check(status);
    
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &status);
    cl_check(status);
    //Compile the program and create the kernel program
    
    size_t vectorSize = sizeof(float) * vectorLength;
    
    cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vectorSize, host_a, NULL);
    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vectorSize, host_b, NULL);
    cl_mem device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vectorSize, NULL, NULL);
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void *)&device_a);
    cl_check(status);

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),(void *)&device_b);
    cl_check(status);
    
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),(void *)&device_c);
    cl_check(status);
    
    status = clSetKernelArg(kernel, 3, sizeof(int),(void *)&vectorLength);
    cl_check(status);
    

    cl_event event= NULL;
    size_t block = 2;
    size_t grid  = ((vectorLength - 1) / block + 1) * block;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &grid, &block, 0, NULL, &event);
    cl_check(status);

    status = clWaitForEvents(1, &event);
    cl_check(status);
    
    clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, vectorSize, host_c, 0, NULL, NULL);
    
    diff = clock() - start;
    printf("Computation takes %lu milliseconds. \n", diff * 1000 / CLOCKS_PER_SEC);
    
    printf("Successfully execute the program.\nChecking the correctness. \n");
    for(int i = 0; i < vectorLength; i++)
    {
        if (host_c[i] != host_validation[i])
        {
            printf("Found incorrect value at %d \n", i);
            return 1;
        }
    }
    
    printf("Correct Solution! \n");
    
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_validation);
    
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    
    return 0;
}