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

#include "CUDA_Activation.h"

namespace Activation {
	template<typename OP, typename DTRes, typename DTArg>
	void Forward_CUDA<OP, DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, DCTX(dctx)) {
		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		VT blend_alpha = 1;
		VT blend_beta = 0;
		VT* d_input;
		VT* d_res;
		size_t sizeOfDataType = sizeof(VT);
		size_t data_buf_size = nr1 * nc1 * sizeOfDataType;

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), data_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), data_buf_size));

		if (res == nullptr) {
			res = DataObjectFactory::create<DTRes>(nr1, nc1, false);
		}

		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(),  data_buf_size, cudaMemcpyHostToDevice));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->data_type, 1, 1, nr1, nc1));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->data_type, 1, 1, nr1, nc1));

		CHECK_CUDNN(cudnnSetActivationDescriptor(ctx->activation_desc, OP::getActivationType(), CUDNN_PROPAGATE_NAN, 0.0));

		CHECK_CUDNN(cudnnActivationForward(ctx->getCuDNNHandle(), ctx->activation_desc, &blend_alpha, ctx->src_tensor_desc,
				d_input, &blend_beta, ctx->dst_tensor_desc, d_res));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, data_buf_size, cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_input));
		CHECK_CUDART(cudaFree(d_res));
	}

	template struct Forward_CUDA<ReLU, DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<ReLU, DenseMatrix<double>, DenseMatrix<double>>;
}

