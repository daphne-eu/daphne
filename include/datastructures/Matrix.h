#ifndef INCLUDE_DATASTRUCTURES_MATRIX_H
#define INCLUDE_DATASTRUCTURES_MATRIX_H

#include <iostream>

class BaseMatrix
{
public:

    BaseMatrix()
    {
    };

    virtual ~BaseMatrix()
    {
    };

    virtual unsigned getRows() const = 0;
    virtual unsigned getCols() const = 0;

    virtual void setSubMat(unsigned startRow, unsigned startCol, BaseMatrix *mat,
                           bool allocSpace = false) = 0;

    virtual BaseMatrix *slice(unsigned beginRow, unsigned beginCol,
                              unsigned endRow, unsigned endCol) const = 0;
};

template <typename T> class AbstractMatrix : public BaseMatrix
{
protected:
    unsigned rows;
    unsigned cols;

public:

    AbstractMatrix() : BaseMatrix()
    {
    };

    AbstractMatrix(unsigned rows, unsigned cols) : rows(rows), cols(cols)
    {
    };

    virtual ~AbstractMatrix()
    {
    };

    unsigned getRows() const override
    {
        return rows;
    };

    unsigned getCols() const override
    {
        return cols;
    };
    virtual T &get(unsigned row, unsigned col) = 0;
    virtual const T &get(unsigned row, unsigned col) const = 0;
    virtual void set(unsigned row, unsigned col, T value) = 0;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const AbstractMatrix<T> &mat)
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

#endif //INCLUDE_DATASTRUCTURES_MATRIX_H