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
		std::cerr << " ----------  biasadd ----------- " << std::endl;
		auto ctx = dctx->getCUDAContext(0);

		using VT = typename DTRes::VT;
		const size_t nr1 = data->getNumRows();
		const size_t nc1 = data->getNumCols();
		const VT blend_alpha = 1;
		const VT blend_beta = 1;
		const VT* d_input = data->getValuesCUDA();
		const VT* d_bias = bias->getValuesCUDA();
//		if(res == nullptr)
//			res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc1, false);
//		VT* d_res = res->getValuesCUDA();
		res = const_cast<DTArg*>(data);
		VT* d_res = const_cast<VT*>(d_input);

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), nr1, nc1, 1, 1));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
		        nr1, nc1, 1, 1));

		CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
				ctx->dst_tensor_desc, d_res));
	}

	template struct BiasAddForward<DenseMatrix<float>, DenseMatrix<float>>;
	template struct BiasAddForward<DenseMatrix<double>, DenseMatrix<double>>;
}
