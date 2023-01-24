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

#include "CUDAHostUtils.h"

extern "C" {
    void cuda_get_device_count(int *device_count) {
        CHECK_CUDART(cudaGetDeviceCount(device_count));
    }

    void cuda_get_mem_info(size_t* available_gpu_mem, size_t* total_gpu_mem) {
        CHECK_CUDART(cudaMemGetInfo(available_gpu_mem, total_gpu_mem));
    }
}