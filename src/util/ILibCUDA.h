/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <runtime/local/vectorized/IVectorizedExecutor.h>
#include "CUDAHostUtils.h"

using fptr_createCUDAvexec = IVectorizedExecutor*(*)();
[[maybe_unused]] static fptr_createCUDAvexec create_cuda_vectorized_executor;

using fptr_destroyCUDAvexec = void(*)(IVectorizedExecutor* vexec);
[[maybe_unused]] static fptr_destroyCUDAvexec destroy_cuda_vectorized_executor;

using fptr_cudaGetDevCount = void(*)(int*);
[[maybe_unused]] static fptr_cudaGetDevCount cuda_get_device_count;

using fptr_cudaGetMemInfo = void(*)(size_t*, size_t*);
[[maybe_unused]] static fptr_cudaGetMemInfo cuda_get_mem_info;

