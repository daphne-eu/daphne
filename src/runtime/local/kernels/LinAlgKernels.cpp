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

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/BaseMatrix.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/kernels/utils.h"

#include <random>

#include <cassert>
#include <cstddef>
#include <cstdint>

// TODO Don't use assertions for checks that could NOT have been done during
// our MLIR-based compilation.

// ****************************************************************************
// Kernel implementations
// ****************************************************************************

template<typename T, template<typename> class uniform_T_distribution>
void randDen(size_t numRows, size_t numCols, int64_t seed, double sparsity,
             T min, T max, BaseMatrix ** out)
{
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");
    assert(min <= max && "min must be <= max");
    assert(sparsity >= 0.0 && sparsity <= 1.0 &&
           "sparsity has to be in the interval [0.0, 1.0]");
    
    DenseMatrix<T> * outDense = DataObjectFactory::create<DenseMatrix<T>>(numRows, numCols, false);
    *out = outDense;

    if (seed == -1) {
        std::random_device rd;
        std::uniform_int_distribution<size_t> seedRnd;
        seed = seedRnd(rd);
    }

    std::mt19937 genVal(seed);
    std::mt19937 genSparse(seed * 3);
    uniform_T_distribution<T> distrVal(min, max);
    std::uniform_real_distribution<double> distrSparse(0.0, 1.0);

    const size_t numValues = numRows * numCols;
    T * values = outDense->getValues();
    for (size_t i = 0; i < numValues; i++) {
        if (distrSparse(genSparse) > sparsity)
            values[i] = T(0);
        else
            values[i] = distrVal(genVal);
    }
}

template<typename T, template<typename> class BinOp>
void elementwiseBinOpDenDenDen(const BaseMatrix * lhs, const BaseMatrix * rhs,
                           BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, lhsDense, lhs);
    dynamic_cast_assert(const DenseMatrix<T> *, rhsDense, rhs);
    const size_t numRows = lhsDense->getNumRows();
    const size_t numCols = lhsDense->getNumCols();
    assert(numRows == rhsDense->getNumRows() && numCols == rhsDense->getNumCols() &&
           "matrix dimensions of lhs and rhs have to match");
    
    DenseMatrix<T> * outDense = DataObjectFactory::create<DenseMatrix<T>>(numRows, numCols, false);
    *out = outDense;
    
    const T * lhsValues = lhsDense->getValues();
    const T * rhsValues = rhsDense->getValues();
    T * outValues = outDense->getValues();
    
    BinOp<T> op;
    const size_t numValues = numRows * numCols;
    for (size_t i = 0; i < numValues; i++)
        outValues[i] = op(lhsValues[i], rhsValues[i]);
}

template<typename T>
void sumDenSca(const BaseMatrix * mat, T * res)
{
    dynamic_cast_assert(const DenseMatrix<T> *, matDense, mat);
    
    const T * values = matDense->getValues();
    
    T agg(0);
    const size_t numValues = mat->getNumRows() * mat->getNumCols();
    for(size_t i = 0; i < numValues; i++)
        agg += values[i];
    
    *res = agg;
}

template<typename T>
void transposeDenDen(const BaseMatrix * in, BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, inDense, in);
    
    const size_t numRows = in->getNumRows();
    const size_t numCols = in->getNumCols();
    
    DenseMatrix<T> * outDense = DataObjectFactory::create<DenseMatrix<T>>(numCols, numRows, false);
    *out = outDense;
    
    const T * inValues = inDense->getValues();
    T * outValues = outDense->getValues();
    for(size_t r = 0, i = 0; r < numRows; r++)
        for(size_t c = 0; c < numCols; c++, i++) {
            size_t j = c * numRows + r;
            outValues[j] = inValues[i];
        }
}

template<typename T>
void setCellDen(BaseMatrix * mat, size_t row, size_t col, T val)
{
    dynamic_cast_assert(DenseMatrix<T> *, matDense, mat);
    matDense->getValues()[row * mat->getNumCols() + col] = val;
}

// ****************************************************************************
// Macros generating functions called from JIT-compiled code
// ****************************************************************************

// TODO Use size_t for numRows/numCols as soon as the IR supports it.
#define MAKE_RAND_DEN(valueTypeName, valueType, distrType) \
    void randDen ## valueTypeName(int64_t numRows, int64_t numCols, int64_t seed, double sparsity, valueType min, valueType max, \
                    BaseMatrix ** out) \
    { \
        randDen<valueType, distrType>(static_cast<size_t>(numRows), static_cast<size_t>(numCols), seed, sparsity, min, max, out); \
    }

#define MAKE_ADD_DENDENDEN(valueTypeName, valueType) \
    void addDenDenDen ## valueTypeName(BaseMatrix * lhs, BaseMatrix * rhs, BaseMatrix ** out) \
    { \
        elementwiseBinOpDenDenDen<valueType, std::plus>(lhs, rhs, out); \
    }

#define MAKE_SUM_DENSCA(valueTypeName, valueType) \
    void sumDenSca ## valueTypeName(BaseMatrix * mat, valueType * res) \
    { \
        sumDenSca<valueType>(mat, res); \
    }

#define MAKE_TRANSPOSE_DENDEN(valueTypeName, valueType) \
    void transposeDenDen ## valueTypeName(BaseMatrix * matIn, BaseMatrix ** matOut) \
    { \
        transposeDenDen<valueType>(matIn, matOut); \
    }

// TODO Use size_t for row/col as soon as the IR supports it.
#define MAKE_SETCELL_DEN(valueTypeName, valueType) \
    void setCellDen ## valueTypeName(BaseMatrix * mat, int64_t row, int64_t col, valueType val) \
    { \
        setCellDen<valueType>(mat, static_cast<size_t>(row), static_cast<size_t>(col), val); \
    } \

// ****************************************************************************
// Functions called from JIT-compiled code
// ****************************************************************************

extern "C"
{
    
    MAKE_RAND_DEN(I64, int64_t, std::uniform_int_distribution);
    MAKE_RAND_DEN(F64, double, std::uniform_real_distribution);
    
    MAKE_ADD_DENDENDEN(I64, int64_t);
    MAKE_ADD_DENDENDEN(F64, double);
    
    MAKE_SUM_DENSCA(I64, int64_t);
    MAKE_SUM_DENSCA(F64, double);
    
    MAKE_TRANSPOSE_DENDEN(I64, int64_t);
    MAKE_TRANSPOSE_DENDEN(F64, double);
    
    MAKE_SETCELL_DEN(I64, int64_t);
    MAKE_SETCELL_DEN(F64, double);

} // extern "C"
