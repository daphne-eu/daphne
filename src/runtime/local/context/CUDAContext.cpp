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

CUDAContext::~CUDAContext() {
	std::cout << "destructing CUDAContext" << std::endl;
	if(cublas_handle) destroy();
}

//std::unique_ptr<CUDAContext> CUDAContext::create(int device_id) {
CUDAContext* CUDAContext::create(int device_id) {
#ifndef NDEBUG
	std::cout << "creating CUDA context..." << std::endl;
#endif
//	std::unique_ptr<CUDAContext> context = std::unique_ptr<CUDAContext>(new CUDAContext(device_id));
	int device_count = -1;
	CHECK_CUDART(cudaGetDeviceCount(&device_count));

	if(device_count < 1) {
		std::cerr << "Not creating requested CUDA context. No cuda devices available." << std::endl;
		return nullptr;
	}

	if(device_id >= device_count) {
		std::cerr << "Requested device ID " << device_id << " >= device count " << device_count << std::endl;
		return nullptr;
	}

	auto* context = new CUDAContext(device_id);
	context->init();
	return context;
}

void CUDAContext::destroy() {
//#ifndef NDEBUG
	std::cout << "Destroying CUDA context..." << std::endl;
//#endif
	CHECK_CUBLAS(cublasDestroy(cublas_handle));
	CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
	CHECK_CUDNN(cudnnDestroy(cudnn_handle));
	CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
	CHECK_CUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
	CHECK_CUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
//	CHECK_CUDART(cudaFree(cublas_workspace));
//	CHECK_CUBLAS(cublasLtDestroy(ltHandle));
}

void CUDAContext::init() {

	CHECK_CUDART(cudaSetDevice(device_id));
	CHECK_CUDART(cudaGetDeviceProperties(&device_properties, device_id));
	std::cout << "Using CUDA device " << device_id << ": " << device_properties.name << std::endl;

	CHECK_CUBLAS(cublasCreate(&cublas_handle));
	CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
	CHECK_CUDNN(cudnnCreate(&cudnn_handle));
	CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));

//	CHECK_CUBLAS(cublasLtCreate(&cublaslt_Handle));
//	CHECK_CUDART(cudaMalloc(&cublas_workspace, cublas_workspace_size));
}
