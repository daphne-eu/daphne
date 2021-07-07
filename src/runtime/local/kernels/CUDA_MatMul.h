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

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul_CUDA {
	static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul_CUDA(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs) {
	MatMul_CUDA<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul_CUDA<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
	static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs) {
		std::cout << "MatMult_CUDA<" << type_name<DenseMatrix<float>>() << "> called" << std::endl;
	}
};

template<>
struct MatMul_CUDA<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
	static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs) {
		std::cout << "MatMult_CUDA<" << type_name<DenseMatrix<double>>() << "> called" << std::endl;
	}
};

#endif //DAPHNE_PROTOTYPE_CUDA_MATMULT_H