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

#include "CUDA_BiasAdd.h"

namespace DNN::CUDA {
	template<typename DTRes, typename DTArg>
	void BiasAddForward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *bias, DCTX(dctx)) {

		auto ctx = dctx->getCUDAContext(0);
		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		const size_t nr2 = bias->getNumRows();
		const size_t nc2 = bias->getNumCols();
		VT blend_alpha = 1;
		VT blend_beta = 1;
		VT* d_input;
		VT* d_bias;
		VT* d_res;
		size_t sizeOfDataType = sizeof(VT);
		size_t data_buf_size = nr1 * nc1 * sizeOfDataType;
		size_t bias_buf_size = nr2 * nc2 * sizeOfDataType;
//		std::cout << "data dims: " << nr1 << "x" << nc1 << " bias dims: " << nr2 << "x" << nc2 << " data_buf_size="
//				<< data_buf_size << " bias_buf_sze=" << bias_buf_size << std::endl;
		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc1, false);

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), data_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_bias), bias_buf_size));
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), data_buf_size));

		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(),  data_buf_size, cudaMemcpyHostToDevice));
		CHECK_CUDART(cudaMemcpy(d_bias, bias->getValues(), bias_buf_size, cudaMemcpyHostToDevice));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), nr1, nc1, 1, 1));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
		        nr1, nc1, 1, 1));

		CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
				ctx->dst_tensor_desc, d_res));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, data_buf_size, cudaMemcpyDeviceToHost));

		CHECK_CUDART(cudaFree(d_input));
		CHECK_CUDART(cudaFree(d_bias));
		CHECK_CUDART(cudaFree(d_res));

	}

	template struct BiasAddForward<DenseMatrix<float>, DenseMatrix<float>>;
	template struct BiasAddForward<DenseMatrix<double>, DenseMatrix<double>>;
}
