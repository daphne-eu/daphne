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

#ifndef DAPHNE_PROTOTYPE_CUDA_MATMULT_H
#define DAPHNE_PROTOTYPE_CUDA_MATMULT_H

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CUDA_HostUtils.h>
#include <runtime/local/kernels/CUDA_Context.h>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul_CUDA {
	static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, const CUDAContext& ctx) = delete;

	[[maybe_unused]] static void applyLT(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, const CUDAContext& ctx) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul_CUDA(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, const CUDAContext& ctx) {
	MatMul_CUDA<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul_CUDA<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
	static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs,
			const CUDAContext& ctx) {
		std::cout << "MatMult_CUDA<" << type_name<DenseMatrix<float>>() << "> called" << std::endl;

		const size_t nr1 = lhs->getNumRows();
		const size_t nc1 = lhs->getNumCols();
		const size_t nr2 = rhs->getNumRows();
		const size_t nc2 = rhs->getNumCols();
		assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);


		const float alpha = 1.0f;
		const float beta = 0.0f;

		// device pointers
		float* d_lhs; float* d_rhs; float* d_res;

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_lhs), nr1 * nc1 * sizeof(float)));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_rhs), nr2 * nc2 * sizeof(float)));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), nr1 * nc2 * sizeof(float)));

		CHECK_CUDART(cudaMemcpy(d_lhs, lhs->getValues(), nr1*nc1*sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_rhs, rhs->getValues(), nr2*nc2*sizeof(float), cudaMemcpyHostToDevice));

		// reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
		CHECK_CUBLAS(cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, &alpha, d_rhs, nc2,
				d_lhs, nc1, &beta, d_res, nc2));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, nr1 * nc2 * sizeof(float), cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_lhs));
		CHECK_CUDART(cudaFree(d_rhs));
		CHECK_CUDART(cudaFree(d_res));
	}
};

template<>
struct MatMul_CUDA<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
	static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs,
						const CUDAContext& ctx) {
		std::cout << "MatMult_CUDA<" << type_name<DenseMatrix<double>>() << "> called" << std::endl;

		const size_t nr1 = lhs->getNumRows();
		const size_t nc1 = lhs->getNumCols();
		const size_t nr2 = rhs->getNumRows();
		const size_t nc2 = rhs->getNumCols();
		assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

		if (res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, true);

	}

	// newer cublasLt API not working atm
	[[maybe_unused]] static void applyLT(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs,
			const CUDAContext& ctx) {
		std::cout << "MatMult_CUDA<" << type_name<DenseMatrix<double>>() << "> called" << std::endl;

		const size_t nr1 = lhs->getNumRows();
		const size_t nc1 = lhs->getNumCols();
		const size_t nr2 = rhs->getNumRows();
		const size_t nc2 = rhs->getNumCols();
		assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, true);

		typedef double InType;
		typedef double OutType;
		int N = 1;
		const double alpha = 1.0f;
		const double beta = 0.0f;
		const size_t& m = nr1;
		const size_t& n = nc2;
		const size_t& k = nc1;
		const size_t& lda = m;
		const size_t& ldb = k;
		const size_t& ldc = m;
		cudaStream_t stream = nullptr;

		void* Adev;
		void* Bdev;
		void* Cdev;

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(OutType)));

		CHECK_CUDART(cudaMemcpy(Adev, lhs->getValues(), nr1*nc1*sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(Bdev, rhs->getValues(), nr2*nc2*sizeof(double), cudaMemcpyHostToDevice));

		cublasLtMatmulDescOpaque_t operationDesc = {};
		cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
		cublasLtMatmulAlgo_t algo = {};

		cublasOperation_t transa = CUBLAS_OP_N;
		cublasOperation_t transb = CUBLAS_OP_N;

		const int32_t algoId = 10;
		const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16; // 5
		const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE; // 1
		const int32_t splitKFactor = 256;

		// create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
		// set the transforms for A and B
		CHECK_CUBLAS(cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
		CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
		CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

		// create matrix descriptors, we are good with the details here so no need to set any extra attributes
		CHECK_CUBLAS(cublasLtMatrixLayoutInit(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
		CHECK_CUBLAS(cublasLtMatrixLayoutInit(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
		CHECK_CUBLAS(cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_64F, m, n, ldc));

		CHECK_CUBLAS(cublasLtMatmulAlgoInit(ctx.getCublasLtHandle(),  //
											CUBLAS_COMPUTE_64F,   // compute
											CUDA_R_64F,   // scale
											CUDA_R_64F,   // A
											CUDA_R_64F,   // B
											CUDA_R_64F,   // C
											CUDA_R_64F,   // D
											algoId,
											&algo));

		CHECK_CUBLAS(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
		CHECK_CUBLAS(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
		CHECK_CUBLAS(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

		CHECK_CUBLAS(cublasLtMatmul(ctx.getCublasLtHandle(),
									&operationDesc,
									&alpha,
									Adev,
									&Adesc,
									Bdev,
									&Bdesc,
									&beta,
									Cdev,
									&Cdesc,
									Cdev,
									&Cdesc,
									&algo,
									ctx.getCublasWorkspacePtr(),
									ctx.getCublasWorkspaceSize(),
									stream));

		CHECK_CUDART(cudaMemcpy(res->getValues(), Cdev, nr1*nc2*sizeof(double), cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(Adev));
		CHECK_CUDART(cudaFree(Bdev));
		CHECK_CUDART(cudaFree(Cdev));
	}
};

#endif //DAPHNE_PROTOTYPE_CUDA_MATMULT_H