/*
 * Copyright 2021 The DAPHNE Consortium
 *
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

#pragma once

#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/context/DaphneContext.h>

#include <cstdint>

#ifdef USE_CUDA
#include <util/ILibCUDA.h>
#include <dlfcn.h>
#endif

#include <iostream>
#include <string>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void createDaphneContext(DaphneContext *& res, uint64_t configPtr) {
    auto config = reinterpret_cast<DaphneUserConfig *>(configPtr);
    res = new DaphneContext(*config);

#ifdef USE_CUDA
    if(res->getUserConfig().use_cuda) {
        void *handle_libCUDAKernels = dlopen("libCUDAKernels.so", RTLD_LAZY);
        if(!handle_libCUDAKernels) {
            throw std::runtime_error("Cannot load libCUDAKernels: " + std::string(dlerror()));

        }
        //reset errors
        dlerror();

        create_cuda_vectorized_executor = reinterpret_cast<fptr_createCUDAvexec>(dlsym(handle_libCUDAKernels,
                                                                                       "cuda_create_vectorized_executor"));

//    auto dlsym_error = std::string(dlerror());

        const char *dlsym_error_cstr = dlerror();
        if(dlsym_error_cstr) {
            auto dlsym_error = std::string(dlsym_error_cstr);
            dlclose(handle_libCUDAKernels);
            throw std::runtime_error("Cannot load symbol cuda_create_vectorized_executor: " + dlsym_error);
        }

        destroy_cuda_vectorized_executor = reinterpret_cast<fptr_destroyCUDAvexec>(dlsym(handle_libCUDAKernels,
                                                                                         "cuda_destroy_vectorized_executor"));

        dlsym_error_cstr = dlerror();
        if(dlsym_error_cstr) {
            auto dlsym_error = std::string(dlsym_error_cstr);
            dlclose(handle_libCUDAKernels);
            throw std::runtime_error("Cannot load symbol cuda_destroy_vectorized_executor: " + dlsym_error);
        }

        cuda_get_device_count = reinterpret_cast<fptr_cudaGetDevCount>(dlsym(handle_libCUDAKernels,
                                                                             "cuda_get_device_count"));
        dlsym_error_cstr = dlerror();
        if(dlsym_error_cstr) {
            auto dlsym_error = std::string(dlsym_error_cstr);
            dlclose(handle_libCUDAKernels);
            throw std::runtime_error("Cannot load symbol cuda_get_device_count: " + dlsym_error);
        }

        cuda_get_mem_info = reinterpret_cast<fptr_cudaGetMemInfo>(dlsym(handle_libCUDAKernels, "cuda_get_mem_info"));
        dlsym_error_cstr = dlerror();
        if(dlsym_error_cstr) {
            auto dlsym_error = std::string(dlsym_error_cstr);
            dlclose(handle_libCUDAKernels);
            throw std::runtime_error("Cannot load symbol cuda_get_mem_info: " + dlsym_error);
        }
    }
#endif
}
