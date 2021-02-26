#ifndef INCLUDE_DATASTRUCTURES_MATRIX_H
#define INCLUDE_DATASTRUCTURES_MATRIX_H

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

#endif //INCLUDE_DATASTRUCTURES_MATRIX_H