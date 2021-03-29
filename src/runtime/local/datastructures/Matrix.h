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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_MATRIX_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_MATRIX_H

#include <cstddef>

/**
 * @brief The base class of all matrix implementations.
 * 
 * All elements of a matrix have the same value type. Rows and columns are
 * addressed starting at zero.
 */
// TODO Could we have the value type as a template parameter here already? Or
// would that cause problems with pure C?
class Matrix
{
protected:
    size_t numRows;
    size_t numCols;

public:

    Matrix(size_t numRows, size_t numCols) :
            numRows(numRows), numCols(numCols)
    {
        // nothing to do
    };

    virtual ~Matrix()
    {
        // nothing to do
    };

    size_t getNumRows() const
    {
        return numRows;
    }

    size_t getNumCols() const
    {
        return numCols;
    }
    
};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_MATRIX_H