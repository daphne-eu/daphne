#ifndef INCLUDE_DATASTRUCTURES_BASEMATRIX_H
#define INCLUDE_DATASTRUCTURES_BASEMATRIX_H

#include <cstddef>

class BaseMatrix
{
protected:
    size_t rows;
    size_t cols;

public:

    BaseMatrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
    };

    virtual ~BaseMatrix()
    {
    };

    size_t getRows() const
    {
        return rows;
    }

    size_t getCols() const
    {
        return cols;
    }

    // TODO Maybe these can be useful later again.
#if 0
    virtual void setSubMat(unsigned startRow, unsigned startCol, BaseMatrix *mat,
                           bool allocSpace = false) = 0;

    virtual BaseMatrix *slice(unsigned beginRow, unsigned beginCol,
                              unsigned endRow, unsigned endCol) const = 0;
#endif
};

#endif //INCLUDE_DATASTRUCTURES_BASEMATRIX_H