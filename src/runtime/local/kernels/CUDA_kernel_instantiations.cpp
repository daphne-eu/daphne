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

extern "C" {
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