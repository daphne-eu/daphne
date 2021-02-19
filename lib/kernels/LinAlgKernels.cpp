#include "datastructures/DenseMatrix.h"
#include "datastructures/Matrix.h"

#include <random>

#include <cassert>

template<typename T> using BinOpFn = std::function<T(T, T)>;

template<typename T>
void denseBinOpElementwise(AbstractMatrix<T> *lhs, AbstractMatrix<T> *rhs,
                           AbstractMatrix<T> **out, BinOpFn<T> op)
{
    assert(lhs->getRows() == rhs->getRows() && lhs->getCols() == rhs->getCols() &&
           "matrix dimensions have to match");
    *out = new DenseMatrix<T>(lhs->getRows(), lhs->getCols());
    for (unsigned r = 0; r < lhs->getRows(); r++) {
        for (unsigned c = 0; c < lhs->getCols(); c++) {
            (*out)->set(r, c, op(lhs->get(r, c), rhs->get(r, c)));
        }
    }
}

template<typename T> bool transpose(BaseMatrix *matIn, BaseMatrix **outIn)
{
    if (auto *mat = dynamic_cast<DenseMatrix<T> *> (matIn)) {
        auto **out = reinterpret_cast<DenseMatrix<T> **> (outIn);
        *out = new DenseMatrix<T>(*mat);
        (*out)->transpose();
        return true;
    }
    return false;
}

template<typename T>
bool binOpMM(BaseMatrix *lhsIn, BaseMatrix *rhsIn, BaseMatrix **outIn,
             BinOpFn<T> op)
{
    if (auto *lhs = dynamic_cast<AbstractMatrix<T> *> (lhsIn)) {
        auto *rhs = dynamic_cast<AbstractMatrix<T> *> (rhsIn);
        assert(rhs && "left and right side have to be of same type");
        auto **out = reinterpret_cast<AbstractMatrix<T> **> (outIn);
        denseBinOpElementwise<T>(lhs, rhs, out, op);
        return true;
    }
    return false;
}


extern "C"
{

    void randRangedMatF64(size_t rows, size_t cols, ssize_t seed, double sparsity,
                          double min, double max, AbstractMatrix<double> **out)
    {
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");
        *out = new DenseMatrix<double>(rows, cols);

        if (seed == -1) {
            std::random_device rd;
            std::uniform_int_distribution<size_t> seedRnd;
            seed = seedRnd(rd);
        }

        std::mt19937 gen(seed);
        std::mt19937 genSparse(seed * 3);
        std::uniform_real_distribution<double> dis(min, max);
        std::uniform_real_distribution<double> sparse(0.0, 1.0);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                if (sparse(genSparse) > sparsity)
                    (*out)->set(r, c, 0.0);
                else
                    (*out)->set(r, c, dis(gen));
            }
        }
    }

    void randRangedMatI64(size_t rows, size_t cols, ssize_t seed, double sparsity,
                          int64_t min, int64_t max, AbstractMatrix<int64_t> **out)
    {
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");
        *out = new DenseMatrix<int64_t>(rows, cols);

        if (seed == -1) {
            std::random_device rd;
            std::uniform_int_distribution<size_t> seedRnd;
            seed = seedRnd(rd);
        }

        std::mt19937 gen(seed);
        std::mt19937 genSparse(seed * 3);
        std::uniform_int_distribution<int64_t> dis(min, max);
        std::uniform_real_distribution<> sparse(0.0, 1.0);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                if (sparse(genSparse) > sparsity)
                    (*out)->set(r, c, 0);
                else
                    (*out)->set(r, c, dis(gen));
            }
        }
    }

    void randMatF64(size_t rows, size_t cols, size_t seed, double sparsity, double min, double max,
                    AbstractMatrix<double> **out)
    {
        randRangedMatF64(rows, cols, seed, sparsity, min, max, out);
    }

    void randMatI64(size_t rows, size_t cols, size_t seed, double sparsity, int64_t min, int64_t max,
                    AbstractMatrix<int64_t> **out)
    {
        randRangedMatI64(rows, cols, seed, sparsity,
                         min,
                         max, out);
    }

    void transpose(BaseMatrix *mat, BaseMatrix **out)
    {
        if (transpose<double>(mat, out) || transpose<float>(mat, out) ||
            transpose<int64_t>(mat, out) || transpose<int32_t>(mat, out)) {
            return;
        }
        assert(false && "Matrix type doesn't match with any implementation");
    }

    void addMM(BaseMatrix *lhs, BaseMatrix *rhs, BaseMatrix **out)
    {
        auto op = [](auto a, auto b)
        {
            return a + b;
        };
        if (binOpMM<double>(lhs, rhs, out, op) || binOpMM<float>(lhs, rhs, out, op) ||
            binOpMM<int64_t>(lhs, rhs, out, op) ||
            binOpMM<int32_t>(lhs, rhs, out, op)) {
            return;
        }
        assert(false && "Matrix types don't match with any implementation");
    }

} // extern "C"
