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

#include "runtime/local/context/CUDAContext.h"

size_t CUDAContext::alloc_count = 0;

using clk = std::chrono::high_resolution_clock;
using sec = std::chrono::duration<double, std::ratio<1>>;

void CUDAContext::destroy() {
    logger->debug("Destroying CUDA context...");
#if 1
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
    CHECK_CUDART(cudaStreamDestroy(cusolver_stream));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bn_tensor_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));

    CHECK_CUDART(cudaFree(cudnn_workspace));
    CHECK_CUDART(cudaFree(scratch_buffer));
#endif

//    CHECK_CUDART(cudaFree(cublas_workspace));
//    CHECK_CUBLAS(cublasLtDestroy(ltHandle));
}

void CUDAContext::init() {
    auto init_start = clk::now();

    CHECK_CUDART(cudaSetDevice(device_id));
    CHECK_CUDART(cudaGetDeviceProperties(&device_properties, device_id));

    size_t available; size_t total;
    cudaMemGetInfo(&available, &total);
    // ToDo: make this a user config item
    float mem_usage = 0.9f;
    mem_budget = total * mem_usage;

    logger->info("Using CUDA device {}: {}\n\tAvailable mem: {} Total mem: {} using {}% -> {}", device_id,
            device_properties.name, available, total, mem_usage * 100, mem_budget);
#if 1
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bn_tensor_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

    CHECK_CUDART(cudaStreamCreateWithFlags(&cusolver_stream, cudaStreamNonBlocking));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle, cusolver_stream));

    getCUDNNWorkspace(64 * 1024 * 1024);


    // allocate 8MB of scratch space (e.g., aggregation tmp buffer)
    CHECK_CUDART(cudaMalloc(&scratch_buffer, 8 * 1048576));
#endif


//    CHECK_CUBLAS(cublasLtCreate(&cublaslt_Handle));
//    CHECK_CUDART(cudaMalloc(&cublas_workspace, cublas_workspace_size));

    auto init_end = clk::now();
	auto init_duration = std::chrono::duration_cast<sec>(init_end - init_start).count();

    logger->info("CUDA context {} init time: {} sec", device_id, init_duration);
}

template<>
cudnnDataType_t CUDAContext::getCUDNNDataType<float>() const {
    return CUDNN_DATA_FLOAT;
}

template<>
cudnnDataType_t CUDAContext::getCUDNNDataType<double>() const {
    return CUDNN_DATA_DOUBLE;
}

template<>
cudaDataType CUDAContext::getCUSparseDataType<float>() const {
    return CUDA_R_32F;
}

template<>
cudaDataType CUDAContext::getCUSparseDataType<double>() const {
    return CUDA_R_64F;
}

void* CUDAContext::getCUDNNWorkspace(size_t size) {
    if (size > cudnn_workspace_size) {
        //ToDo: don't leak memory here if workspace is already allocated
        logger->debug("Allocating cuDNN workspace of size {} bytes", size);
        CHECK_CUDART(cudaMalloc(&cudnn_workspace, size));
        cudnn_workspace_size = size;
    }
    else {
        logger->debug("Not allocating cuDNN conv workspace of size {} bytes", size);
    }

    return cudnn_workspace;
}

std::unique_ptr<IContext> CUDAContext::createCudaContext(int device_id) {

    auto ctx = std::unique_ptr<CUDAContext>(new CUDAContext(device_id));
    ctx->logger = spdlog::get("runtime::cuda");
    ctx->logger->debug("runtime::cuda logger set for context {}", device_id);

    int device_count = -1;
    CHECK_CUDART(cudaGetDeviceCount(&device_count));

    if(device_count < 1) {
        ctx->logger->warn("Not creating requested CUDA context. No cuda devices available.");
        return nullptr;
    }

    if(device_id >= device_count) {
        ctx->logger->warn("Requested device ID {} >= device count {}", device_id, device_count);
        return nullptr;
    }

    ctx->init();
//    return reinterpret_cast<std::unique_ptr<IContext> &&>(ctx);
    return ctx;
}

std::shared_ptr<std::byte> CUDAContext::malloc(size_t size, bool zero, size_t& id) {
    id = alloc_count++;
    std::byte* dev_ptr;
    CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&dev_ptr), size));
    allocations.emplace(id, std::shared_ptr<std::byte>(dev_ptr, CudaDeleter<std::byte>()));

    if(zero)
        CHECK_CUDART(cudaMemset(dev_ptr, 0, size));
    return allocations.at(id);
}

void CUDAContext::free(size_t id) {
    // ToDo: handle reuse
    CHECK_CUDART(cudaFree(allocations.at(id).get()));
    allocations.erase(id);
}

int CUDAContext::getMaxNumThreads() {
    return device_properties.maxThreadsPerBlock;
}
