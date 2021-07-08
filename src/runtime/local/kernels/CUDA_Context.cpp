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

#include "CUDA_Context.h"

CUDAContext::~CUDAContext() = default;

CUDAContext * CUDAContext::create(DaphneUserConfig *config) {
	std::cout << "creating CUDA context..." << std::endl;
	config->context = std::make_unique<CUDAContext>();
	config->context->cublas_workspace_size = config->cublas_workspace_size;
	assert(config->cuda_devices.size() == 1 && "Multi device support not implemented");
	config->context->device_id = config->cuda_devices.front();
	config->context->init();
	return config->context.get();
}

void CUDAContext::destroy() {
	std::cout << "destroying CUDA context..." << std::endl;
	CHECK_CUBLAS(cublasDestroy(cublas_handle));
//	CHECK_CUDART(cudaFree(cublas_workspace));
//	CHECK_CUBLAS(cublasLtDestroy(ltHandle));
}

void CUDAContext::init() {
	CHECK_CUDART(cudaSetDevice(device_id));
	CHECK_CUBLAS(cublasCreate(&cublas_handle));
//	CHECK_CUBLAS(cublasLtCreate(&cublaslt_Handle));
//	CHECK_CUDART(cudaMalloc(&cublas_workspace, cublas_workspace_size));
}
