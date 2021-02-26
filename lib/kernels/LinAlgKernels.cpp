#include "datastructures/DenseMatrix.h"
#include "datastructures/Matrix.h"
#include "utils.h"

#include <random>

#include <cassert>
#include <cstddef>
#include <cstdint>

// TODO Get rid of the virtual get/set calls.
// TODO Don't use assertions for checks that could NOT have been done during
// our MLIR-based compilation.

// ****************************************************************************
// Kernel implementations
// ****************************************************************************

template<typename T, template<typename> class uniform_T_distribution>
void randDen(size_t rows, size_t cols, int64_t seed, double sparsity,
             T min, T max, BaseMatrix ** out)
{
    assert(rows > 0 && "rows must be > 0");
    assert(cols > 0 && "cols must be > 0");
    assert(min <= max && "min must be <= max");
    assert(sparsity >= 0.0 && sparsity <= 1.0 &&
           "sparsity has to be in the interval [0.0, 1.0]");
    
    DenseMatrix<T> * outDense = new DenseMatrix<T>(rows, cols);
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

    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++) {
            if (distrSparse(genSparse) > sparsity)
                outDense->set(r, c, T(0));
            else
                outDense->set(r, c, distrVal(genVal));
        }
}

template<typename T, template<typename> class BinOp>
void elementwiseBinOpDenDenDen(const BaseMatrix * lhs, const BaseMatrix * rhs,
                           BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, lhsDense, lhs);
    dynamic_cast_assert(const DenseMatrix<T> *, rhsDense, rhs);
    assert(lhsDense->getRows() == rhsDense->getRows() && lhsDense->getCols() == rhsDense->getCols() &&
           "matrix dimensions of lhs and rhs have to match");
    
    DenseMatrix<T> * outDense = new DenseMatrix<T>(lhsDense->getRows(), lhsDense->getCols());
    *out = outDense;
    
    BinOp<T> op;
    for (size_t r = 0; r < lhsDense->getRows(); r++)
        for (size_t c = 0; c < lhsDense->getCols(); c++)
            outDense->set(r, c, op(lhsDense->get(r, c), rhsDense->get(r, c)));
}

template<typename T>
void transposeDenDen(const BaseMatrix * in, BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, inDense, in);
    
    DenseMatrix<T> * outDense = new DenseMatrix<T>(*inDense);
    *out = outDense;
    
    outDense->transpose();
}

template<typename T>
void setCellDen(BaseMatrix * mat, size_t row, size_t col, T val)
{
    dynamic_cast_assert(DenseMatrix<T> *, matDense, mat);
    matDense->set(row, col, val);
}

// ****************************************************************************
// Macros generating functions called from JIT-compiled code
// ****************************************************************************

// TODO Use size_t for rows/cols as soon as the IR supports it.
#define MAKE_RAND_DEN(valueTypeName, valueType, distrType) \
    void randDen ## valueTypeName(int64_t rows, int64_t cols, int64_t seed, double sparsity, valueType min, valueType max, \
                    BaseMatrix ** out) \
    { \
        randDen<valueType, distrType>(static_cast<size_t>(rows), static_cast<size_t>(cols), seed, sparsity, min, max, out); \
    }

#define MAKE_ADD_DENDENDEN(valueTypeName, valueType) \
    void addDenDenDen ## valueTypeName(BaseMatrix * lhs, BaseMatrix * rhs, BaseMatrix ** out) \
    { \
        elementwiseBinOpDenDenDen<valueType, std::plus>(lhs, rhs, out); \
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
    
    MAKE_TRANSPOSE_DENDEN(I64, int64_t);
    MAKE_TRANSPOSE_DENDEN(F64, double);
    
    MAKE_SETCELL_DEN(I64, int64_t);
    MAKE_SETCELL_DEN(F64, double);

} // extern "C"
