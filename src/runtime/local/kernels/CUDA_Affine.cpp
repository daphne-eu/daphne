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

#include "CUDA_Affine.h"

template<typename T>
static void launch_cublas_gemm(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2, const T* alpha, const T* beta,
							   T* d_lhs, T* d_rhs, T* d_res);

template<>
[[maybe_unused]] void launch_cublas_gemm<float>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
													   const float* alpha,	const float* beta, float* d_lhs, float* d_rhs, float* d_res) {
	CHECK_CUBLAS(cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
							 nc1, beta, d_res, nc2));
}

template<>
[[maybe_unused]] void launch_cublas_gemm<double>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
														const double* alpha, const double* beta, double* d_lhs, double* d_rhs, double* d_res) {
	CHECK_CUBLAS(cublasDgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
							 nc1, beta, d_res, nc2));
}

namespace Affine {
	template<typename DTRes, typename DTArg>
	void Forward_CUDA<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *weights, const DTArg *bias, DCTX(dctx)) {
//		matMul_CUDA<DTRes, DTArg, DTArg>(res, data, weights, dctx);
		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		const size_t nr2 = weights->getNumRows();
		const size_t nc2 = weights->getNumCols();
		VT blend_alpha = 1;
		VT blend_beta = 0;
		VT* d_input;
		VT* d_weights;
		VT* d_bias;
		VT* d_res;
		size_t sizeOfDataType = sizeof(VT);
		size_t data_buf_size = data->getNumRows() * data->getNumCols() * sizeOfDataType;

		assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), data_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_weights), nr2 * nc2 * sizeOfDataType));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), nr1 * nc2 * sizeOfDataType));

		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(),  data_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_weights, weights->getValues(),  nr2 * nc2 * sizeOfDataType, cudaMemcpyHostToDevice));

		// reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
		launch_cublas_gemm<VT>(*ctx, nr1, nc1, nc2, &blend_alpha, &blend_beta, d_weights, d_input, d_res);

		if(bias) {
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_bias), bias->getNumCols() * sizeOfDataType));
			CHECK_CUDART(cudaMemcpy(d_bias, bias->getValues(), bias->getNumCols() * sizeOfDataType, cudaMemcpyHostToDevice));

			CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), 1,
												   bias->getNumCols(), 1, 1));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
					nr1, nc2, 1, 1));
			blend_beta = 1;
			CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
									   ctx->dst_tensor_desc, d_res));
			CHECK_CUDART(cudaFree(d_bias));
		}

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, nr1 * nc2 * sizeOfDataType, cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_input));
		CHECK_CUDART(cudaFree(d_weights));
		CHECK_CUDART(cudaFree(d_res));

	}

	template struct Forward_CUDA<DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<DenseMatrix<double>, DenseMatrix<double>>;
}
