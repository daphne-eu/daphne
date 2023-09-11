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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cudnn.h>
#include <cusolverDn.h>

#include <iostream>
#include <memory>

#define CHECK_CUDART(call)                                                \
  do {                                                                    \
    cudaError_t status = call;                                            \
    if (status != cudaSuccess) {                                          \
        throw std::runtime_error(fmt::format("(CUDART) returned: {} ({}:{}:{}())", \
                cudaGetErrorString(status), __FILE__, __LINE__, __func__));\
    }                                                                     \
  } while (0)

#define CHECK_CUBLAS(call)                                                \
  do {                                                                    \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
        throw std::runtime_error(fmt::format("(CUBLAS) returned: {} ({}:{}:{}())", \
                status, __FILE__, __LINE__, __func__));\
    }                                                                     \
  } while (0)

#define CHECK_CUSPARSE(call)                                                 \
    do {                                                                      \
        cusparseStatus_t status = call;                                        \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        throw std::runtime_error(fmt::format("(CUSPARSE) returned: {} ({}:{}:{}())",  \
                status, __FILE__, __LINE__, __func__)); \
        }                                                                           \
    } while (0)

#define CHECK_CUDNN(call)                                                  \
  do {                                                                    \
    cudnnStatus_t status = call;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                                 \
        throw std::runtime_error(fmt::format("(CUDNN) returned: {} ({}:{}:{}())", \
                status, __FILE__, __LINE__, __func__));\
    }                                                                     \
  } while (0)

#define CHECK_CUSOLVER(call)                                                   \
  do {                                                                    \
    cusolverStatus_t status = call;                                         \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                \
        throw std::runtime_error(fmt::format("(CUSOLVER) returned: {} ({}:{}:{}())", \
                status, __FILE__, __LINE__, __func__));\
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

template<typename T>
struct CudaDeleter {
    void operator()(T* dev_ptr) const { del(dev_ptr); };
    static void del(T* dev_ptr) {
        cudaFree(reinterpret_cast<void*>(dev_ptr));
    }
};

template<typename T>
void cuda_deleter(T* dev_ptr) { CudaDeleter<T>::del(dev_ptr); }

template<typename T>
using CudaUniquePtr [[maybe_unused]] = std::unique_ptr<T, decltype(&cuda_deleter<T>)>;

static inline uint32_t divup(unsigned n, unsigned div)
{
    return (n + div - 1) / div;
}

template<typename VT>
struct smem_calc /*: std::unary_function<int, int> */ {
    int operator()(int i) const { return sizeof(VT) * i; }

};


static uint32_t nextPow2(uint32_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}