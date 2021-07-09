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

#include "CUDA_MatMul.h"

template<typename T>
void launch_cublas_gemm(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2, const T* alpha, const T* beta,
		T* d_lhs, T* d_rhs, T* d_res);

template<>
void launch_cublas_gemm<float>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2, const float* alpha,
	const float* beta, float* d_lhs, float* d_rhs, float* d_res) {
	CHECK_CUBLAS(cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2,
			d_lhs, nc1, beta, d_res, nc2));
}

template<>
void launch_cublas_gemm<double>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2, const double* alpha,
		const double* beta, double* d_lhs, double* d_rhs, double* d_res) {
	CHECK_CUBLAS(cublasDgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2,
			d_lhs, nc1, beta, d_res, nc2));
}

template<class DTRes, class DTLhs, class DTRhs>
void MatMul_CUDA<DTRes, DTLhs, DTRhs>::apply(DTRes*& res, const DTLhs* lhs, const DTRhs* rhs,
		const CUDAContext& ctx) {
	std::cout << "MatMult_CUDA<" << type_name<typename DTRes::value_type>() << "> called" << std::endl;

	const size_t nr1 = lhs->getNumRows();
	const size_t nc1 = lhs->getNumCols();
	const size_t nr2 = rhs->getNumRows();
	const size_t nc2 = rhs->getNumCols();
	assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

	if(res == nullptr)
		res = DataObjectFactory::create<DTRes>(nr1, nc2, false);


	const typename DTRes::value_type alpha = 1.0f;
	const typename DTRes::value_type beta = 0.0f;

	// device pointers
	typename DTRes::value_type* d_lhs;
	typename DTRes::value_type* d_rhs;
	typename DTRes::value_type* d_res;

	size_t sizeOfDataType = sizeof(typename DTRes::value_type);

	CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_lhs), nr1 * nc1 * sizeOfDataType));
	CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_rhs), nr2 * nc2 * sizeOfDataType));
	CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), nr1 * nc2 * sizeOfDataType));

	CHECK_CUDART(cudaMemcpy(d_lhs, lhs->getValues(), nr1*nc1*sizeOfDataType, cudaMemcpyHostToDevice));
	CHECK_CUDART(cudaMemcpy(d_rhs, rhs->getValues(), nr2*nc2*sizeOfDataType, cudaMemcpyHostToDevice));

	// reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
	launch_cublas_gemm<typename DTRes::value_type>(ctx, nr1, nc1, nc2, &alpha, &beta, d_lhs, d_rhs, d_res);

	CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, nr1 * nc2 * sizeOfDataType, cudaMemcpyDeviceToHost));

	CHECK_CUDART(cudaFree(d_lhs));
	CHECK_CUDART(cudaFree(d_rhs));
	CHECK_CUDART(cudaFree(d_res));
}

// explicit instantiations to satisfy linker
template struct MatMul_CUDA<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
template struct MatMul_CUDA<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;