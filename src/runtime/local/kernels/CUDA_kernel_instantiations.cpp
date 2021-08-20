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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/CUDA_MatMul.h>
#include <runtime/local/kernels/CUDA_InitContext.h>
#include <runtime/local/kernels/CUDA_Pooling.h>

extern "C" {

	// -----------------------------------------------------------------------------------------------------------------
	void _avgPoolForward_CUDA__DenseMatrix_float__size_t__size_t__DenseMatrix_float__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t
			(DenseMatrix<float>** res, size_t* res_h, size_t* res_w, const DenseMatrix<float>* data,
			 const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
			 const size_t pool_h, const size_t pool_w, const size_t stride_h, const size_t stride_w,
			 const size_t pad_h, const size_t pad_w, DCTX(ctx)) {
		Pooling::Forward_CUDA<Pooling::AVG, DenseMatrix<float>, DenseMatrix<float>>::apply(*res, *res_h, *res_w,
				data, batch_size, num_channels, img_h, img_w, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, ctx);
	}

	void _avgPoolForward_CUDA__DenseMatrix_double__size_t__size_t__DenseMatrix_double__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t
	(DenseMatrix<double>** res, size_t* res_h, size_t* res_w, const DenseMatrix<double>* data,
	 const size_t batch_size, size_t num_channels,
	 const size_t img_h, const size_t img_w, const size_t pool_h,
			 const size_t pool_w, const size_t stride_h, const size_t stride_w,
	 const size_t pad_h, const size_t pad_w,
			 DCTX(ctx)) {
		Pooling::Forward_CUDA<Pooling::AVG, DenseMatrix<double>, DenseMatrix<double>>::apply(*res, *res_h, *res_w,data, batch_size,
																							 num_channels, img_h, img_w, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, ctx);
	}

	// -----------------------------------------------------------------------------------------------------------------
	void _maxPoolForward_CUDA__DenseMatrix_float__size_t__size_t__DenseMatrix_float__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t
			(DenseMatrix<float>** res, size_t* res_h, size_t* res_w, const DenseMatrix<float>* data, const size_t batch_size,
			 const size_t num_channels, const size_t img_h, const size_t img_w, const size_t pool_h, const size_t pool_w,
			 const size_t stride_h, const size_t stride_w, const size_t pad_h, const size_t pad_w, DCTX(ctx)) {
		Pooling::Forward_CUDA<Pooling::MAX, DenseMatrix<float>, DenseMatrix<float>>::apply(*res, *res_h, *res_w,
				data, batch_size, num_channels, img_h, img_w, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, ctx);
	}

	void _maxPoolForward_CUDA__DenseMatrix_double__size_t__size_t__DenseMatrix_double__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t__size_t
			(DenseMatrix<double>** res, size_t* res_h, size_t* res_w, const DenseMatrix<double>* data,
			 const size_t batch_size, const size_t num_channels,
	 		const size_t img_h, const size_t img_w,
			 const size_t pool_h, const size_t pool_w, const size_t stride_h, uint32_t stride_w, const size_t pad_h,
	 		const size_t pad_w, DCTX(ctx)) {
		Pooling::Forward_CUDA<Pooling::MAX, DenseMatrix<double>, DenseMatrix<double>>::apply(*res, *res_h, *res_w,
				data, batch_size, num_channels, img_h, img_w, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, ctx);
	}

	void matMul_CUDA__DenseMatrix_float__DenseMatrix_float__DenseMatrix_float(DenseMatrix<float>** res,
			const DenseMatrix<float>* lhs, const DenseMatrix<float>* rhs, DCTX(ctx)) {
		MatMul_CUDA<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>::apply(*res, lhs, rhs, ctx);
	}

	void matMul_CUDA__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(DenseMatrix<double>** res,
			const DenseMatrix<double>* lhs, const DenseMatrix<double>* rhs, DCTX(ctx)) {
		MatMul_CUDA<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>::apply(*res, lhs, rhs, ctx);
	}

	void _initCUDAContext(DCTX(ctx)) {
		initCUDAContext(ctx);
	}
}