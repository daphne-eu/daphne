#include "datastructures/BaseMatrix.h"
#include "datastructures/DenseMatrix.h"
#include "utils.h"

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

    const size_t numCells = rows * cols;
    T * cells = outDense->getCells();
    for (size_t i = 0; i < numCells; i++) {
        if (distrSparse(genSparse) > sparsity)
            cells[i] = T(0);
        else
            cells[i] = distrVal(genVal);
    }
}

template<typename T, template<typename> class BinOp>
void elementwiseBinOpDenDenDen(const BaseMatrix * lhs, const BaseMatrix * rhs,
                           BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, lhsDense, lhs);
    dynamic_cast_assert(const DenseMatrix<T> *, rhsDense, rhs);
    const size_t rows = lhsDense->getRows();
    const size_t cols = lhsDense->getCols();
    assert(rows == rhsDense->getRows() && cols == rhsDense->getCols() &&
           "matrix dimensions of lhs and rhs have to match");
    
    DenseMatrix<T> * outDense = new DenseMatrix<T>(rows, cols);
    *out = outDense;
    
    const T * lhsCells = lhsDense->getCells();
    const T * rhsCells = rhsDense->getCells();
    T * outCells = outDense->getCells();
    
    BinOp<T> op;
    const size_t numCells = rows * cols;
    for (size_t i = 0; i < numCells; i++)
        outCells[i] = op(lhsCells[i], rhsCells[i]);
}

template<typename T>
void sumDenSca(const BaseMatrix * mat, T * res)
{
    dynamic_cast_assert(const DenseMatrix<T> *, matDense, mat);
    
    const T * cells = matDense->getCells();
    
    T agg(0);
    const size_t numCells = mat->getRows() * mat->getCols();
    for(size_t i = 0; i < numCells; i++)
        agg += cells[i];
    
    *res = agg;
}

template<typename T>
void transposeDenDen(const BaseMatrix * in, BaseMatrix ** out)
{
    dynamic_cast_assert(const DenseMatrix<T> *, inDense, in);
    
    const size_t rows = in->getRows();
    const size_t cols = in->getCols();
    
    DenseMatrix<T> * outDense = new DenseMatrix<T>(cols, rows);
    *out = outDense;
    
    const T * inCells = inDense->getCells();
    T * outCells = outDense->getCells();
    for(size_t r = 0, i = 0; r < rows; r++)
        for(size_t c = 0; c < cols; c++, i++) {
            size_t j = c * rows + r;
            outCells[i] = inCells[j];
            outCells[j] = inCells[i];
        }
}

template<typename T>
void setCellDen(BaseMatrix * mat, size_t row, size_t col, T val)
{
    dynamic_cast_assert(DenseMatrix<T> *, matDense, mat);
    matDense->getCells()[row * mat->getCols() + col] = val;
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
