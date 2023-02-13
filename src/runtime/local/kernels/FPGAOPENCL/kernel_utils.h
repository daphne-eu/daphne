#ifndef SRC_RUNTIME_LOCAL_FPGAOPENCL_KERNEL_UTILS_H
#define SRC_RUNTIME_LOCAL_FPGAOPENCL_KERNEL_UTILS_H


#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"

using namespace aocl_utils;

void *acl_aligned_malloc(size_t size);

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d);
#endif
