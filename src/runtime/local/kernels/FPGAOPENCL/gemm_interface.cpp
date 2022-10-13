// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.


// This file is modified from /glob/development-tools/versions/fpgasupportstack/a10/1.2.1/intelFPGA_pro/hld/examples_aoc/matrix_mult/host/src/main.cpp

#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <runtime/local/context/FPGAContext.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/FPGAOPENCL/gemm_interface.h>



// Parameters of the systolic array
#define II   32
#define JJ   32
#define KK   32
#define III  14
#define JJJ  16
#define KKK  16

using namespace aocl_utils;

#define TYPE float

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define DPRINTF(...)     \
    printf(__VA_ARGS__); \
    fflush(stdout);

#define NUM_QUEUES_TO_CREATE    6
#define NUM_KERNELS_TO_CREATE   6

#define CHECK(status)                                       \
    if (status != CL_SUCCESS) {                             \
        printf("error %d in line %d.\n", status, __LINE__); \
        exit(1);                                            \
    }

#define ACL_ALIGNMENT 64
void *acl_aligned_malloc(size_t size) {
    void *result = NULL;
    posix_memalign(&result, ACL_ALIGNMENT, size);
    return result;
}

void cleanup() {}

const char *kernel_name[] = {
    "kernel_A_loader",
    "kernel_B_loader",
    "kernel_unloader_WAIT_FINISH",
    "kernel_A_feeder",
    "kernel_B_feeder",
    "kernel_Out"
};

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d) {
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    start_d = (double)1.0e-9 * start;
    end_d = (double)1.0e-9 * end;
    //return (double)(end-start);
    return (double)1.0e-9 * (end - start); // nanoseconds to seconds
}


int sgemm(const float *A, const float *B, float *C, const int OUTERMOST_I, const int OUTERMOST_J, const int OUTERMOST_K, DCTX(ctx)) {
    const int TOTAL_I = III * II * OUTERMOST_I;
    const int TOTAL_J = JJJ * JJ * OUTERMOST_J;
    const int TOTAL_K = KKK * KK * OUTERMOST_K;
    
    long int num_elem_A = (long int)TOTAL_I*TOTAL_K;
    long int num_elem_B = (long int)TOTAL_K*TOTAL_J;
    long int num_elem_C = (long int)TOTAL_I*TOTAL_J;

    float *serialized_A, *serialized_B;
    if ((serialized_A = (float *)acl_aligned_malloc(num_elem_A * sizeof(float))) == NULL) {
        perror("Failed malloc of matrix serialized_A");
    }
    if ((serialized_B = (float *)acl_aligned_malloc(num_elem_B * sizeof(float))) == NULL) {
        perror("Failed malloc of matrix serialized_A");
    }

    // Serialize A
    long int addr = 0;
    for (int i = 0; i < TOTAL_I; i++)
        for (int k = 0; k < TOTAL_K; k++) {
            serialized_A[addr++] = A[k + i*TOTAL_K];
        }
    // Serialize B
    addr = 0;
    for (int j = 0; j < TOTAL_J; j++)
        for (int k = 0; k < TOTAL_K; k++) {
            serialized_B[addr++] = B[j+k*TOTAL_J];
        }


    cl_int status;
    auto fctx = ctx->getFPGAContext(0);    

    //----------------------------------------------
    // Create command queues
    //---------------------------------------------

    cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE + 1]; // extra queue for reading buffer D

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute on
    for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
        //fDPRINTF(stdout,"cmdQueue i = %d\n", i);
        cmdQueue[i] = clCreateCommandQueue(
            fctx->context,
            fctx->devices[0],
            CL_QUEUE_PROFILING_ENABLE,
            &status);
        CHECK(status);
    }

    //fDPRINTF(stdout,"cmdQueue i = %d, a queue for reading the C buffer\n", i);
    cmdQueue[NUM_QUEUES_TO_CREATE] = clCreateCommandQueue(
        fctx->context,
        fctx->devices[0],
        CL_QUEUE_PROFILING_ENABLE,
        &status);
    CHECK(status);

    //----------------------------------------------
    // Create device buffers
    //----------------------------------------------
    cl_mem input_A_buf;
    cl_mem input_B_buf;
    cl_mem output_C_buf;
#ifndef NDEBUG
    DPRINTF("\n===== Host-CPU transferring W and X to the FPGA device global memory (DDR4) via PCIe ======\n\n");
#endif
    input_A_buf = clCreateBuffer(
        fctx->context,
        CL_MEM_READ_ONLY,
        num_elem_A * sizeof(cl_float),
        NULL,
        &status);
    CHECK(status);

    input_B_buf = clCreateBuffer(
        fctx->context,
        CL_MEM_READ_ONLY,
        num_elem_B * sizeof(cl_float),
        NULL,
        &status);
    CHECK(status);

    output_C_buf = clCreateBuffer(
        fctx->context,
        CL_MEM_WRITE_ONLY,
        num_elem_C * sizeof(cl_float),
        NULL,
        &status);
    CHECK(status);

    //----------------------------------------------
    // Write host data to device buffers
    //----------------------------------------------
    // blocking writes
    status = clEnqueueWriteBuffer(
        cmdQueue[0],
        input_A_buf,
        CL_TRUE,
        0,
        num_elem_A * sizeof(cl_float),
        serialized_A,
        0,
        NULL,
        NULL);
    CHECK(status);

    status = clEnqueueWriteBuffer(
        cmdQueue[1],
        input_B_buf,
        CL_TRUE,
        0,
        num_elem_B * sizeof(cl_float),
        serialized_B,
        0,
        NULL,
        NULL);
    CHECK(status);

    //----------------------------------------------
    // Create the program from binaries
    //----------------------------------------------
    //DPRINTF("\n===== Host-CPU setting up OpenCL program and kernels ======\n\n");

    cl_program program;
    size_t binary_length;
    const unsigned char *binary;

    fflush(stdout);
    // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
    char *aocx_file = getenv("BITSTREAM");
    FILE *fp = fopen(aocx_file, "rb");

    if (fp == NULL) {
        DPRINTF("Failed to open the AOCX file (fopen).\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    binary_length = ftell(fp);
    binary = (unsigned char *)malloc(sizeof(unsigned char) * binary_length);
    assert(binary && "Malloc failed");
    rewind(fp);

    if (fread((void *)binary, binary_length, 1, fp) == 0) {
        DPRINTF("Failed to read from the AOCX file (fread).\n");
        return -1;
    }
    fclose(fp);

    //DPRINTF("Create program with binary\n");
    // Create a program using clCreateProgramWithBinary()
    program = clCreateProgramWithBinary(
        fctx->context,
        1,
        fctx->devices,
        &binary_length,
        (const unsigned char **)&binary,
        &status,
        NULL);
    CHECK(status);

    //----------------------------------------------
    // Create the kernel
    //----------------------------------------------
    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char log[128 * 1024] = {0};
        clGetProgramBuildInfo(
		program, 
		fctx->devices[0], 
		CL_PROGRAM_BUILD_LOG, 128 * 1024, log, NULL);
        CHECK(status);
    }

    cl_kernel kernel[NUM_KERNELS_TO_CREATE];

    for (int j = 0; j < NUM_KERNELS_TO_CREATE; j++) {
        kernel[j] = clCreateKernel(program, (const char *)kernel_name[j], &status);
        CHECK(status);
    }
#ifndef NDEBUG
    DPRINTF("All kernels created\n");
#endif
    // A_loader
    status = clSetKernelArg(
        kernel[0],
        0,
        sizeof(int),
	&TOTAL_K);
    CHECK(status);
    status = clSetKernelArg(
        kernel[0],
        1,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[0],
        2,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);
    status = clSetKernelArg(
        kernel[0],
        3,
        sizeof(cl_mem),
	&input_A_buf);
    CHECK(status);
    // B_loader
    status = clSetKernelArg(
        kernel[1],
        0,
        sizeof(int),
	&TOTAL_K);
    CHECK(status);
    status = clSetKernelArg(
        kernel[1],
        1,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[1],
        2,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);
    status = clSetKernelArg(
        kernel[1],
        3,
        sizeof(cl_mem),
	&input_B_buf);
    CHECK(status);
    // unloader
    status = clSetKernelArg(
        kernel[2],
        0,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[2],
        1,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);
    status = clSetKernelArg(
        kernel[2],
        2,
        sizeof(cl_mem),
	&output_C_buf);
    CHECK(status);
    // A_feeder
    status = clSetKernelArg(
        kernel[3],
        0,
        sizeof(int),
	&TOTAL_K);
    CHECK(status);
    status = clSetKernelArg(
        kernel[3],
        1,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[3],
        2,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);
    // B_feeder
    status = clSetKernelArg(
        kernel[4],
        0,
        sizeof(int),
	&TOTAL_K);
    CHECK(status);
    status = clSetKernelArg(
        kernel[4],
        1,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[4],
        2,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);
    // Out
    status = clSetKernelArg(
        kernel[5],
        0,
        sizeof(int),
	&TOTAL_K);
    CHECK(status);
    status = clSetKernelArg(
        kernel[5],
        1,
        sizeof(int),
	&TOTAL_I);
    CHECK(status);
    status = clSetKernelArg(
        kernel[5],
        2,
        sizeof(int),
	&TOTAL_J);
    CHECK(status);

    //----------------------------------------------
    // Configure the work-item structure (using only tasks atm)
    //----------------------------------------------

    // Define the number of threads that will be created
    // as well as the number of work groups
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    //----------------------------------------------
    // Enqueue the kernel for execution
    //----------------------------------------------

    // all kernels are always tasks
    globalWorkSize[0] = 1;
    localWorkSize[0] = 1;

    cl_event kernel_exec_event[NUM_KERNELS_TO_CREATE];

#ifndef NDEBUG
    DPRINTF("\n===== Host-CPU enqeuing the OpenCL kernels to the FPGA device ======\n\n");
#endif
    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        // Alternatively, can use clEnqueueTaskKernel
#ifndef NDEBUG
        DPRINTF("clEnqueueNDRangeKernel[%d]: %s!\n", i, kernel_name[i]);
#endif
	status = clEnqueueNDRangeKernel(
            cmdQueue[i],
            kernel[i],
            1,
            NULL,
            globalWorkSize,
            localWorkSize,
            0,
            NULL,
            &kernel_exec_event[i]);
        CHECK(status);
    }
#ifndef NDEBUG
    DPRINTF(" *** FPGA execution started!\n");
#endif    
    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        status = clFlush(cmdQueue[i]);
        CHECK(status);
    }

    for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
#ifndef NDEBUG
        DPRINTF("cmd queue: %d\n", i);
#endif    
        fflush(stdout);
        status = clFinish(cmdQueue[i]);
        CHECK(status);
    }
#ifndef NDEBUG
    DPRINTF(" *** FPGA execution finished!\n");
    DPRINTF("\n\n");
//#endif    
 
    double k_start_time[NUM_KERNELS_TO_CREATE];
    double k_end_time[NUM_KERNELS_TO_CREATE];
    double k_exec_time[NUM_KERNELS_TO_CREATE];
    double max_time = 0;
    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        k_exec_time[i] = compute_kernel_execution_time(kernel_exec_event[i], k_start_time[i], k_end_time[i]);
        if (k_exec_time[i] > max_time) {
            max_time = k_exec_time[i];
        }
    }
//#ifndef NDEBUG
    DPRINTF("Time taken: %lf sec\n\n", max_time);

    printf("\n===== Reporting measured throughput ======\n\n");
//#endif    
    double k_earliest_start_time = k_start_time[0];
    double k_latest_end_time = k_end_time[0];
    
    for (int i = 1; i < NUM_KERNELS_TO_CREATE; i++) {
        if (k_start_time[i] < k_earliest_start_time)
            k_earliest_start_time = k_start_time[i];

        if (k_end_time[i] > k_latest_end_time)
            k_latest_end_time = k_end_time[i];
    }

    // IMPORTANT: we care about the finish time of drain_C, once data is drained we are done
    k_latest_end_time = k_end_time[NUM_KERNELS_TO_CREATE - 1];

    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        printf("  Kernel execution time on FPGA: %s, \n   \t\t\t\t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", kernel_name[i], k_exec_time[i], k_start_time[i], k_end_time[i]);
    }
//#endif
 
    double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;
//#ifndef NDEBUG
    printf("\n");
    printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);
    printf("  Unloader kernels end time\t\t= %.5f s\n", k_latest_end_time);
    printf("  FPGA GEMM exec time\t\t= %.5f s\n", k_overall_exec_time);

    // multiplied by 1.0e-9 to get G-FLOPs
    printf("\n");

    double num_operations = (double)2.0 * (TOTAL_K) * (double)(TOTAL_I) * (double)(TOTAL_J);

    printf("  # operations = %.0f\n", num_operations );
    printf("  Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);

    DPRINTF("\n===== Host-CPU transferring result matrix C from the FPGA device global memory (DDR4) via PCIe ======\n\n");
#endif
    // Read the results back from the device, blocking read
    float *serialized_Z;
    if ((serialized_Z = (float *)acl_aligned_malloc(num_elem_C * sizeof(float))) == NULL) {
        perror("Failed malloc of matrix serialized_Z");
    }

    clEnqueueReadBuffer(
        //cmdQueue[KID_DRAIN_MAT_C],
        cmdQueue[NUM_KERNELS_TO_CREATE], // using a special queue for reading buffer C
        output_C_buf,
        CL_TRUE,
        0,
        num_elem_C * sizeof(cl_float),
        serialized_Z,
        0,
        NULL,
        NULL);
    CHECK(status);

    // Deserialize Z
    addr = 0;
    for (int i = 0; i < TOTAL_I; i++)
        for (int j = 0; j < TOTAL_J; j++) {
            C[j + i*TOTAL_J] = serialized_Z[addr++];
        }
    return 0;
}

