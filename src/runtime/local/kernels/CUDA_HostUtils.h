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

#ifndef DAPHNE_PROTOTYPE_CUDAHOSTUTILS_H
#define DAPHNE_PROTOTYPE_CUDAHOSTUTILS_H

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    CUresult status = call;                                               \
    if (status != CUDA_SUCCESS) {                                         \
      const char* str;                                                    \
      cuGetErrorName(status, &str);                                       \
      std::cout << "(CUDA) returned: " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
    }                                                                     \
  } while (0)

#define CHECK_CUDART(call)                                                \
  do {                                                                    \
    cudaError_t status = call;                                            \
    if (status != cudaSuccess) {                                          \
      std::cout << "(CUDART) returned: " << cudaGetErrorString(status);    \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
    }                                                                     \
  } while (0)

#define CHECK_CUBLAS(call)                                                \
  do {                                                                    \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      std::cout << "(CUBLAS) returned " << status;    \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
    }                                                                     \
  } while (0)

#include <string_view>

template <typename T>
static constexpr auto type_name() noexcept {
	std::string_view name, prefix, suffix;
#ifdef __clang__
	name = __PRETTY_FUNCTION__;
  prefix = "auto type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
	name = __PRETTY_FUNCTION__;
	prefix = "constexpr auto type_name() [with T = ";
	suffix = "]";
#elif defined(_MSC_VER)
	name = __FUNCSIG__;
  prefix = "auto __cdecl type_name<";
  suffix = ">(void) noexcept";
#endif
	name.remove_prefix(prefix.size());
	name.remove_suffix(suffix.size());
	return name;
}

#endif //DAPHNE_PROTOTYPE_CUDAHOSTUTILS_H
