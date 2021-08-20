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

#include "CUDA_BatchNorm.h"

namespace BatchNorm {
	template<typename DTRes, typename DTArg>
	void ForwardTest_CUDA<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *gamma, const DTArg *beta,
			const DTArg *ema_mean, const DTArg *ema_var, const typename DTArg::VT eps, DCTX(dctx))
	{
		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		VT blend_alpha = 1.0;
		VT blend_beta = 0.0;
		VT* d_input;
		VT* d_gamma;
		VT* d_beta;
		VT* d_ema_mean;
		VT* d_ema_var;
		VT* d_res;
		size_t num_channels = gamma->getNumRows();
		size_t sizeOfDataType = sizeof(VT);
		size_t data_buf_size = nr1 * nc1 * sizeOfDataType;
		size_t aux_buf_size = num_channels * sizeOfDataType;

		size_t HW = nc1 / num_channels;
		auto H = static_cast<size_t>(std::sqrt(HW));
//		std::cout << "N=" << nr1 << " C=" << num_channels << " H=" << H << " eps: " << eps << std::endl;
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), data_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), data_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_gamma), aux_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_beta), aux_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_ema_mean), aux_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_ema_var), aux_buf_size));

		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(), data_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_gamma, gamma->getValues(), aux_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_beta, beta->getValues(),  aux_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_ema_mean, ema_mean->getValues(),  aux_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_ema_var, ema_var->getValues(),  aux_buf_size, cudaMemcpyHostToDevice));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->data_type, nr1, num_channels, H, H));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->data_type, nr1, num_channels, H, H));

		if (res == nullptr) {
			res = DataObjectFactory::create<DTRes>(nr1, nc1, false);
		}

		CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(ctx->bn_tensor_desc, ctx->src_tensor_desc, ctx->bn_mode));
		CHECK_CUDNN(cudnnBatchNormalizationForwardInference(ctx->getCuDNNHandle(), ctx->bn_mode, &blend_alpha,
					&blend_beta, ctx->src_tensor_desc, d_input, ctx->dst_tensor_desc, d_res, ctx->bn_tensor_desc,
					d_gamma, d_beta, d_ema_mean, d_ema_var, eps));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, data_buf_size, cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_input));
		CHECK_CUDART(cudaFree(d_gamma));
		CHECK_CUDART(cudaFree(d_beta));
		CHECK_CUDART(cudaFree(d_ema_mean));
		CHECK_CUDART(cudaFree(d_ema_var));
		CHECK_CUDART(cudaFree(d_res));
	}

	template struct ForwardTest_CUDA<DenseMatrix<float>, DenseMatrix<float>>;
	template struct ForwardTest_CUDA<DenseMatrix<double>, DenseMatrix<double>>;
}

