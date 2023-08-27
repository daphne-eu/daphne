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

#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/kernels/CUDA/HostUtils.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <map>

class CUDAContext final : public IContext {
    int device_id = -1;
    size_t mem_budget = 0;

    cudaDeviceProp device_properties{};

    cublasHandle_t cublas_handle = nullptr;

    cusparseHandle_t cusparse_handle = nullptr;

    // cuDNN API
    cudnnHandle_t cudnn_handle{};

    cusolverDnHandle_t cusolver_handle{};
    cudaStream_t cusolver_stream{};

    // preallocate 64MB
    size_t cudnn_workspace_size{};
    void* cudnn_workspace{};
    
    std::map<size_t, std::shared_ptr<std::byte>> allocations;
    static size_t alloc_count;

    explicit CUDAContext(int id) : device_id(id) {
        logger = spdlog::get("runtime::cuda");
    }
    
    void init();
    
public:
    CUDAContext() = delete;
    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    ~CUDAContext() = default;

    void destroy() override;
    static std::unique_ptr<IContext> createCudaContext(int id);

    [[nodiscard]] cublasHandle_t getCublasHandle() const { return cublas_handle; }
    [[nodiscard]] cusparseHandle_t getCusparseHandle() const { return cusparse_handle; }

    [[nodiscard]] const cudaDeviceProp* getDeviceProperties() const { return &device_properties; }
    [[nodiscard]] cudnnHandle_t  getCUDNNHandle() const { return cudnn_handle; }
    [[nodiscard]] cusolverDnHandle_t getCUSOLVERHandle() const { return cusolver_handle; }
    cudaStream_t getCuSolverStream() { return cusolver_stream; }

    template<class T>
    [[nodiscard]] cudnnDataType_t getCUDNNDataType() const;

    template<class T>
    [[nodiscard]] cudaDataType getCUSparseDataType() const;

    void* getCUDNNWorkspace(size_t size);

    [[nodiscard]] size_t getMemBudget() const { return mem_budget; }
    int getMaxNumThreads();
    static CUDAContext* get(DaphneContext* ctx, size_t id) { return dynamic_cast<CUDAContext*>(ctx->getCUDAContext(id)); }

    std::shared_ptr<std::byte> malloc(size_t size, bool zero, size_t& id);

    void free(size_t id);

    template<typename T>
    static void debugPrintCUDABuffer(const CUDAContext& ctx, std::string_view title, const T* data, size_t num_items) {
        std::vector<T> tmp(num_items);
        CHECK_CUDART(cudaMemcpy(tmp.data(), data, num_items * sizeof(T), cudaMemcpyDeviceToHost));
        auto out = fmt::memory_buffer();
        fmt::format_to(std::back_inserter(out),"{} \n", title);
        fmt::format_to(std::back_inserter(out), fmt::join(tmp, ", "));
        ctx.logger->debug(out);
    }

    int conv_algorithm = -1;
    cudnnPoolingDescriptor_t pooling_desc{};
    cudnnTensorDescriptor_t src_tensor_desc{}, dst_tensor_desc{}, bn_tensor_desc{};
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    cudnnFilterDescriptor_t filter_desc{};
    cudnnActivationDescriptor_t  activation_desc{};
    cudnnConvolutionDescriptor_t conv_desc{};
    cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL;

    // A block size of 256 works well in many cases.
    // Putting it here to avoid hard coding things elsewhere.
    const uint32_t default_block_size = 256;

    // cuda runtime logger
    std::shared_ptr<spdlog::logger> logger;
};
