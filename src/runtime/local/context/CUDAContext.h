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

#ifndef DAPHNE_PROTOTYPE_CUDACONTEXT_H
#define DAPHNE_PROTOTYPE_CUDACONTEXT_H

#pragma once

//#include <api/cli/DaphneUserConfig.h>
#include "IContext.h"
#include <runtime/local/kernels/CUDA_HostUtils.h>
//#include <cublasLt.h>

#include <cassert>
#include <iostream>
#include <memory>

class CUDAContext : public IContext {
    int device_id = -1;

    cudaDeviceProp device_properties{};

    cublasHandle_t cublas_handle = nullptr;

    cusparseHandle_t cusparse_handle = nullptr;

    // cuDNN API
    cudnnHandle_t cudnn_handle{};

    // preallocate 64MB
    size_t cudnn_workspace_size{};
    void* cudnn_workspace{};

    // cublasLt API
//    cublasLtHandle_t cublaslt_Handle = nullptr;
//    void* cublas_workspace{};
//    size_t cublas_workspace_size{};

    explicit CUDAContext(int id) : device_id(id) { }
public:
    CUDAContext() = delete;
    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    ~CUDAContext() = default;

    void destroy() override;
    static std::unique_ptr<IContext> createCudaContext(int id);

    [[nodiscard]] cublasHandle_t getCublasHandle() const { return cublas_handle; }
    [[nodiscard]] cusparseHandle_t getCusparseHandle() const { return cusparse_handle; }

//    [[nodiscard]] cublasLtHandle_t getCublasLtHandle() const { return cublaslt_Handle; }
//    [[nodiscard]] void* getCublasWorkspacePtr() const { return cublas_workspace; }
//    [[nodiscard]] size_t getCublasWorkspaceSize() const { return cublas_workspace_size; }
    [[nodiscard]] const cudaDeviceProp* getDeviceProperties() const { return &device_properties; }
    [[nodiscard]] cudnnHandle_t  getCUDNNHandle() const { return cudnn_handle; }

    template<class T>
    [[nodiscard]] cudnnDataType_t getCUDNNDataType() const;

    template<class T>
    [[nodiscard]] cudaDataType getCUSparseDataType() const;

    void* getCUDNNWorkspace(size_t size);

    int conv_algorithm = -1;
    cudnnPoolingDescriptor_t pooling_desc{};
    cudnnTensorDescriptor_t src_tensor_desc{}, dst_tensor_desc{}, bn_tensor_desc{};
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    cudnnFilterDescriptor_t filter_desc{};
    cudnnActivationDescriptor_t  activation_desc{};
    cudnnConvolutionDescriptor_t conv_desc{};
    cudnnFilterDescriptor_t filterDesc{};
    cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL;


private:
    void init();
};

#endif //DAPHNE_PROTOTYPE_CUDACONTEXT_H