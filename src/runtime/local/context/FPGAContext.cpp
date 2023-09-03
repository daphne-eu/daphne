/*
 * Copyright 2021 The DAPHNE Consortium
 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"
#include "runtime/local/context/FPGAContext.h"
//#include <cstdio>
//#include <cstdlib>
//#include <cstring>
//#include <fstream>
//#include <iomanip>
//#include <iostream>
//#include <math.h>
//#include <sstream>
//#include <stdint.h>
//#include <stdio.h>

using namespace std;
using namespace aocl_utils;

#define DPRINTF(...)     \
    printf(__VA_ARGS__); \
    fflush(stdout);

#define CHECK(status)                                       \
    if (status != CL_SUCCESS) {                             \
        throw std::runtime_error(fmt::format("error {} in line {}", status, __LINE__)); \
    }

void FPGAContext::destroy() {
    spdlog::debug("Destroying FPGA context...");
}

void FPGAContext::init() {
    spdlog::debug("creating FPGA context...");
    spdlog::debug("\n===== Host-CPU setting up the OpenCL platform and device ======\n\n");
    unsigned int buf_uint;
    cl_int status;
    char buffer[4096];
    int device_found = 0;

    // Use clGetPlatformIDs() to retrieve the  number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    spdlog::debug("Number of platforms = {}\n", numPlatforms);

    // Allocate enough space for each platform
    platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));

    spdlog::debug("Allocated space for Platform\n");

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK(status);
#ifndef NDEBUG
    DPRINTF("Filled in platforms\n");
    DPRINTF("Initializing IDs\n");
#endif
    for (int i = 0; i < (int)numPlatforms; i++) {
        status = clGetDeviceIDs(platforms[i],
                                CL_DEVICE_TYPE_ALL,
                                maxDevices,
                                devices,
                                &numDevices);

        if (status == CL_SUCCESS) {
            clGetPlatformInfo(platforms[i],
                              CL_PLATFORM_NAME,
                              4096,
                              buffer,
                              NULL);
#if defined(ALTERA_CL)
            if (strstr(buffer, "Altera") != NULL) {
                device_found = 1;
            }
//            DPRINTF("%s\n", buffer);
#elif defined(NVIDIA_CL)
            if (strstr(buffer, "NVIDIA") != NULL) {
                device_found = 1;
            }
#else
            if (strstr(buffer, "Intel") != NULL) {
                device_found = 1;
            }
#endif
#ifndef NDEBUG
            DPRINTF("Platform found : %s\n", buffer);
#endif
	    device_found = 1;
        }
    }
    if (!device_found) {
        DPRINTF("failed to find a OpenCL device\n");
        exit(-1);
    }
#ifndef NDEBUG
    DPRINTF("Total number of devices: %d", numDevices);
    for (unsigned int i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_NAME,
                        4096,
                        buffer,
                        NULL);
        DPRINTF("\nDevice Name: %s\n", buffer);
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_VENDOR,
                        4096,
                        buffer,
                        NULL);
        DPRINTF("Device Vendor: %s\n", buffer);
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        DPRINTF("Device Computing Units: %u\n", buf_uint);
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        DPRINTF("Global Memory Size: %li\n", *((unsigned long*)buffer));
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        DPRINTF("Global Memory Allocation Size: %li\n\n", *((unsigned long*)buffer));
    }
#endif
    //----------------------------------------------
    // Create a context
#ifndef NDEBUG
    DPRINTF("\n===== Host-CPU setting up the OpenCL command queues ======\n\n");
#endif
    context = clCreateContext(
        NULL,
        1,
        devices,
        NULL,
        NULL,
        &status);
    CHECK(status);
}

std::unique_ptr<IContext> FPGAContext::createFpgaContext(int device_id) {

/*    	if(FPGAContext::numDevices < 1) {
        std::cerr << "Not creating requested FPGA context. No FPGA devices available." << std::endl;
        return nullptr;
    }

    if(device_id >= (int)numDevices) {
        std::cerr << "Requested device ID " << device_id << " >= device count "<<std::endl;// << device_count << std::endl;
        return nullptr;
    }
*/
    auto ctx = std::unique_ptr<FPGAContext>(new FPGAContext(device_id));
    ctx->init();

    return ctx;
}


