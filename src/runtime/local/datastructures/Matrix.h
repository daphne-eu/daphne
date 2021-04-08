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

#include <runtime/local/datastructures/Structure.h>

#include <cstddef>

/**
 * @brief The base class of all matrix implementations.
 * 
 * All elements of a matrix have the same value type. Rows and columns are
 * addressed starting at zero.
 */
template<typename ValueType>
class Matrix : public Structure
{

protected:

    Matrix(size_t numRows, size_t numCols) :
            Structure(numRows, numCols)
    {
        // nothing to do
    };

    virtual ~Matrix()
    {
        // nothing to do
    };

};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_MATRIX_H