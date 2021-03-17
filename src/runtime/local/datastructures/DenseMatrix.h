#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H

#include "runtime/local/datastructures/BaseMatrix.h"

#include <functional>
#include <iostream>

#include <cassert>
#include <cmath>
#include <cstddef>

template <typename T>
class DenseMatrix : public BaseMatrix
{
private:
    T * cells;

public:
    DenseMatrix(size_t rows, size_t cols);
    virtual ~DenseMatrix();

    const T * getCells() const
    {
        return cells;
    };

    T * getCells()
    {
        return cells;
    }

    // TODO Maybe these can be useful later again.
#if 0
    DenseMatrix &operator=(DenseMatrix &other) = delete;
    DenseMatrix &operator=(const DenseMatrix &&other);

    void setSubMat(unsigned startRow, unsigned startCol, BaseMatrix *mat,
                   bool allocSpace) override;
    BaseMatrix *slice(unsigned beginRow, unsigned beginCol, unsigned endRow,
                      unsigned endCol) const override;

    void resize(unsigned rows, unsigned cols);

    void fill(T value);
#endif
};

template <typename T>
DenseMatrix<T>::DenseMatrix(size_t rows, size_t cols)
: BaseMatrix(rows, cols), cells(new T[rows * cols])
{
}

// TODO Maybe these can be useful later again.
#if 0
template <typename T>
DenseMatrix<T>::DenseMatrix(const DenseMatrix<T> &other)
: DenseMatrix(other.rows, other.cols)
{
    // TODO check for same size
    std::copy(other.cells, other.cells + rows * cols, cells);
}
#endif

// TODO Maybe these can be useful later again.
#if 0
template <typename T>
DenseMatrix<T> &DenseMatrix<T>::operator=(const DenseMatrix<T> &&other)
{
    AbstractMatrix<T>::rows = other.rows;
    other.rows = 0;
    AbstractMatrix<T>::cols = other.cols;
    other.cols = 0;
    cells = other.cells;
    other.cells = nullptr;
    transposed = other.transposed;
    other.transposed = false;
    return *this;
}
#endif

template <typename T>
DenseMatrix<T>::~DenseMatrix()
{
    delete[] cells;
}

// TODO Maybe these can be useful later again.
#if 0
template <typename T>
void DenseMatrix<T>::setSubMat(unsigned startRow, unsigned startCol,
                               BaseMatrix *mat, bool allocSpace)
{
    if (allocSpace) {
        auto neededRows = std::max(getRows(), startRow + mat->getRows());
        auto neededCols = std::max(getCols(), startCol + mat->getCols());
        resize(neededRows, neededCols);
    }
    assert(startRow + mat->getRows() <= getRows() &&
           "Sub-Matrix has to fit in matrix");
    assert(startCol + mat->getCols() <= getCols() &&
           "Sub-Matrix has to fit in matrix");
    auto *castMat = dynamic_cast<AbstractMatrix<T> *> (mat);
    assert(castMat && "Sub-Matrix hast to have the same element type");

    for (auto r = 0u; r < mat->getRows(); r++) {
        for (auto c = 0u; c < mat->getCols(); c++) {
            set(startRow + r, startCol + c, castMat->get(r, c));
        }
    }
}

template <typename T>
BaseMatrix *DenseMatrix<T>::slice(unsigned beginRow, unsigned beginCol,
                                  unsigned endRow, unsigned endCol) const
{
    assert(endRow <= getRows() && "Slice-Matrix has to be contained in matrix");
    assert(endCol <= getCols() && "Slice-Matrix has to be contained in matrix");
    assert(beginRow <= endRow && "Begin has to be smaller than end index");
    assert(beginCol <= endCol && "Begin has to be smaller than end index");
    auto numRows = endRow - beginRow;
    auto numCols = endCol - beginCol;
    auto *out = new DenseMatrix<T>(numRows, numCols);

    for (auto r = 0u; r < numRows; r++) {
        for (auto c = 0u; c < numCols; c++) {
            out->set(r, c, get(beginRow + r, beginCol + c));
        }
    }
    return out;
}

template <typename T>
void DenseMatrix<T>::resize(unsigned rows, unsigned cols)
{
    if (transposed)
        std::swap(rows, cols);
    if (rows == AbstractMatrix<T>::rows && cols == AbstractMatrix<T>::cols)
        return;
    auto *newArr = new T[rows * cols];
    // don't use getRows(), we want raw rows (transpose would change what we get)
    for (auto r = 0u; r < AbstractMatrix<T>::rows; r++) {
        for (auto c = 0u; c < AbstractMatrix<T>::cols; c++) {
            newArr[r * cols + c] = cells[r * AbstractMatrix<T>::cols + c];
        }
    }
    delete[] cells;
    cells = newArr;
    AbstractMatrix<T>::rows = rows;
    AbstractMatrix<T>::cols = cols;
}

template <typename T> void DenseMatrix<T>::fill(T value)
{
    std::fill_n(cells, getRows() * getCols(), value);
}
#endif

template <typename T>
std::ostream &operator<<(std::ostream &os, const DenseMatrix<T> &mat)
{
    const T * cells = mat.getCells();

    size_t i = 0;

    os << "Matrix(rows = " << mat.getRows() << ", cols = " << mat.getCols()
            << ")\n[";
    for (unsigned r = 0; r < mat.getRows(); r++) {
        if (r != 0)
            os << " ";
        os << "[";
        for (unsigned c = 0; c < mat.getCols(); c++) {
            os << cells[i++];
            if (c < mat.getCols() - 1) {
                os << " ";
            }
        }
        os << "]";
        if (r < mat.getRows() - 1) {
            os << "\n";
        }
    }
    return os << "]\n";
}

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H