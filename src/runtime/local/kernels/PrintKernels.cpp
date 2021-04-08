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

#include "runtime/local/datastructures/Matrix.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/kernels/utils.h"

#include <iostream>

#include <cassert>
#include <cstdint>

// ****************************************************************************
// Kernel implementations
// ****************************************************************************

template<typename T>
void printSca(T val)
{
    std::cout << val << std::endl;
}

template<typename T>
void printDen(const Matrix<T> * mat)
{
    dynamic_cast_assert(const DenseMatrix<T> *, matDense, mat);
    std::cout << *matDense;
}

// ****************************************************************************
// Macros generating functions called from JIT-compiled code
// ****************************************************************************

#define MAKE_PRINT_SCA(valueTypeName, valueType) \
    void printSca ## valueTypeName(valueType sca) \
    { \
        printSca<valueType>(sca); \
    }
    
#define MAKE_PRINT_DEN(valueTypeName, valueType) \
    void printDen ## valueTypeName(Matrix<valueType> * mat) \
    { \
        printDen<valueType>(mat); \
    }

// ****************************************************************************
// Functions called from JIT-compiled code
// ****************************************************************************

extern "C"
{
    
    MAKE_PRINT_SCA(I64, int64_t);
    MAKE_PRINT_SCA(F64, double);
    
    MAKE_PRINT_DEN(I64, int64_t);
    MAKE_PRINT_DEN(F64, double);
    
} // extern "C"