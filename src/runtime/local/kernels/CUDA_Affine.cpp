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
							   const T* d_lhs, const T* d_rhs, T* d_res);

template<>
[[maybe_unused]] void launch_cublas_gemm<float>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
		const float* alpha,	const float* beta, const float* d_lhs, const float* d_rhs, float* d_res) {
	CHECK_CUBLAS(cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
			nc1, beta, d_res, nc2));
}

template<>
[[maybe_unused]] void launch_cublas_gemm<double>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
		const double* alpha, const double* beta, const double* d_lhs, const double* d_rhs, double* d_res) {
	CHECK_CUBLAS(cublasDgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
							 nc1, beta, d_res, nc2));
}

namespace Affine {
	template<typename DTRes, typename DTArg>
	void Forward_CUDA<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *weights, const DTArg *bias, DCTX(dctx)) {
		std::cerr << " ----------  affine ----------- " << std::endl;
		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		const size_t nr2 = weights->getNumRows();
		const size_t nc2 = weights->getNumCols();
		const VT blend_alpha = 1;
		VT blend_beta = 0;
		const VT* d_input = data->getValuesCUDA();
		const VT* d_weights = weights->getValuesCUDA();

		assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");

		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false, ALLOCATION_TYPE::CUDA_ALLOC);
		VT* d_res = res->getValuesCUDA();

		// reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
		launch_cublas_gemm<VT>(*ctx, nr1, nc1, nc2, &blend_alpha, &blend_beta, d_weights, d_input, d_res);

		if(bias) {
			std::cout << " bias vector: " << *bias << std::endl;
			std::cout << "bias dims: " << bias->getNumRows() << "x" << bias->getNumCols() << std::endl;
			std::cout << "data dims: " << data->getNumRows() << "x" << data->getNumCols() << std::endl;
			std::cout << "res dims: " << res->getNumRows() << "x" << res->getNumCols() << std::endl;
//return;
			const VT* d_bias = bias->getValuesCUDA();
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
			        1, bias->getNumCols(), 1, 1));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
					nr1, nc2, 1, 1));
			blend_beta = 1;
//			return;
			CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
					ctx->dst_tensor_desc, d_res));
		}
	}

	template struct Forward_CUDA<DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<DenseMatrix<double>, DenseMatrix<double>>;
}
