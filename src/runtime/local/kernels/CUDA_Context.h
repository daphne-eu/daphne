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

#ifndef DAPHNE_PROTOTYPE_CUDA_CONTEXT_H
#define DAPHNE_PROTOTYPE_CUDA_CONTEXT_H

#pragma once

#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/kernels/CUDA_HostUtils.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <cassert>
#include <iostream>
#include <memory>

struct DaphneUserConfig;

class CUDAContext {
	int device_id = -1;

	cublasHandle_t cublas_handle = nullptr;

	// cublasLt API
	cublasLtHandle_t cublaslt_Handle = nullptr;
	void* cublas_workspace{};
	size_t cublas_workspace_size{};

public:
	static CUDAContext* create(DaphneUserConfig* config);

	void destroy();

	[[nodiscard]] cublasLtHandle_t getCublasLtHandle() const { return cublaslt_Handle; }
	[[nodiscard]] cublasHandle_t getCublasHandle() const { return cublas_handle; }
	[[nodiscard]] void* getCublasWorkspacePtr() const { return cublas_workspace; }
	[[nodiscard]] size_t getCublasWorkspaceSize() const { return cublas_workspace_size; }

	~CUDAContext();

private:
	void init();
};

#endif //DAPHNE_PROTOTYPE_CUDA_CONTEXT_H