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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_BASEMATRIX_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_BASEMATRIX_H

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

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_BASEMATRIX_H