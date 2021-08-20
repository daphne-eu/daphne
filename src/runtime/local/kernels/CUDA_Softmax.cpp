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

#include "CUDA_Softmax.h"

namespace Softmax {

	template<typename DTRes, typename DTArg>
	void Forward_CUDA<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, DCTX(dctx)) {

		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		int n = data->getNumRows();
		int d = data->getNumCols();
		VT* d_input;
		VT* d_res;
		size_t sizeOfDataType = sizeof(VT);

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), n*d*sizeOfDataType));
		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(), n*d*sizeOfDataType, cudaMemcpyHostToDevice));

		VT alpha = 1;
		VT beta = 0;

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->data_type, n, d, 1, 1));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->data_type, n, d, 1, 1));


		if (res == nullptr) {
			res = DataObjectFactory::create<DTRes>(n,d, false);
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), static_cast<unsigned long>(n) * d * sizeOfDataType));
		}
		else
//			resize(n*c*h*w, dstData);
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), n * d * sizeOfDataType));

		CHECK_CUDNN(cudnnSoftmaxForward(ctx->getCuDNNHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha, ctx->src_tensor_desc, d_input, &beta, ctx->dst_tensor_desc, d_res));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, n*d*sizeOfDataType, cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_input));
		CHECK_CUDART(cudaFree(d_res));
	}

	template struct Forward_CUDA<DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<DenseMatrix<double>, DenseMatrix<double>>;
}

