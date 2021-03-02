#include "datastructures/BaseMatrix.h"
#include "datastructures/DenseMatrix.h"
#include "utils.h"

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
void printDen(const BaseMatrix * mat)
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
    void printDen ## valueTypeName(BaseMatrix * mat) \
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