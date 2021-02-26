#ifndef INCLUDE_DATASTRUCTURES_DENSEMATRIX_H
#define INCLUDE_DATASTRUCTURES_DENSEMATRIX_H

#include "datastructures/Matrix.h"

#include <iostream>

#include <cassert>
#include <cmath>
#include <functional>

template <typename T> class DenseMatrix : public AbstractMatrix<T>
{
private:
    T *cells;
    bool transposed;

public:
    DenseMatrix();
    DenseMatrix(unsigned rows, unsigned cols);
    DenseMatrix(const DenseMatrix &other);
    virtual ~DenseMatrix();

    DenseMatrix &operator=(DenseMatrix &other) = delete;
    DenseMatrix &operator=(const DenseMatrix &&other);

    unsigned getRows() const override;
    unsigned getCols() const override;

    T &get(unsigned row, unsigned col) override;
    const T &get(unsigned row, unsigned col) const override;
    void set(unsigned row, unsigned col, T value) override;

    void setSubMat(unsigned startRow, unsigned startCol, BaseMatrix *mat,
                   bool allocSpace) override;
    BaseMatrix *slice(unsigned beginRow, unsigned beginCol, unsigned endRow,
                      unsigned endCol) const override;

    void resize(unsigned rows, unsigned cols);

    void fill(T value);
    void transpose();
};

template <typename T>
DenseMatrix<T>::DenseMatrix()
: AbstractMatrix<T>(), cells(nullptr), transposed(false)
{
}

template <typename T>
DenseMatrix<T>::DenseMatrix(unsigned rows, unsigned cols)
: AbstractMatrix<T>(rows, cols), cells(new T[rows * cols]),
transposed(false)
{
}

template <typename T>
DenseMatrix<T>::DenseMatrix(const DenseMatrix<T> &other)
: DenseMatrix(other.rows, other.cols)
{
    transposed = other.transposed;
    std::copy(other.cells, other.cells + getRows() * getCols(), cells);
}

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

template <typename T> DenseMatrix<T>::~DenseMatrix()
{
    delete[] cells;
}

template <typename T> unsigned DenseMatrix<T>::getRows() const
{
    return transposed ? AbstractMatrix<T>::cols : AbstractMatrix<T>::rows;
}

template <typename T> unsigned DenseMatrix<T>::getCols() const
{
    return transposed ? AbstractMatrix<T>::rows : AbstractMatrix<T>::cols;
}

template <typename T> T &DenseMatrix<T>::get(unsigned row, unsigned col)
{
    if (transposed)
        return cells[col * getRows() + row];
    else
        return cells[row * getCols() + col];
}

template <typename T>
const T &DenseMatrix<T>::get(unsigned row, unsigned col) const
{
    if (transposed)
        return cells[col * getRows() + row];
    else
        return cells[row * getCols() + col];
}

template <typename T>
void DenseMatrix<T>::set(unsigned row, unsigned col, T value)
{
    get(row, col) = value;
}

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

template <typename T> void DenseMatrix<T>::transpose()
{
    transposed = !transposed;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const DenseMatrix<T> &mat)
{
    os << "Matrix(rows = " << mat.getRows() << ", cols = " << mat.getCols()
            << ")\n[";
    for (unsigned r = 0; r < mat.getRows(); r++) {
        if (r != 0)
            os << " ";
        os << "[";
        for (unsigned c = 0; c < mat.getCols(); c++) {
            os << mat.get(r, c);
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

#endif //INCLUDE_DATASTRUCTURES_DENSEMATRIX_H