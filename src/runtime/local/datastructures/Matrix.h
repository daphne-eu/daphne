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

public:

    virtual ~Matrix()
    {
        // nothing to do
    };

    template<typename NewValueType>
    using WithValueType = Matrix<NewValueType>;
    
    /**
     * @brief The common type of all values in this matrix.
     */
    using VT = ValueType;
    
    /**
     * @brief Returns the value at the given coordinates in this matrix.
     * 
     * Expect this method to be inefficient and fall back to it only if not
     * avoidable. Use specific access methods of the particular sub-class
     * whenever possible.
     * 
     * @param rowIdx
     * @param colIdx
     * @return The value at coordinates `rowIdx`, `colIdx`.
     */
    virtual ValueType get(size_t rowIdx, size_t colIdx) const = 0;
    
    /**
     * @brief Set the value at the given coordinates in this matrix.
     * 
     * Assumes that this matrix is in a structurally valid state. What exactly
     * this means depends on the particular sub-class. Usually, a matrix might
     * not be structurally valid if its underlying arrays are uninitialized or
     * during the time it is populated by a kernel. After each call to this
     * method, this matrix will be in a structurally valid state again.
     * 
     * You should refrain from mixing `set`, `append`, and direct access to the
     * underlying arrays to populate a matrix.
     * 
     * Expect this method to be inefficient and fall back to it only if not
     * avoidable. Use specific access methods of the particular sub-class
     * whenever possible.
     * 
     * @param rowIdx
     * @param colIdx
     * @param value
     */
    virtual void set(size_t rowIdx, size_t colIdx, ValueType value) = 0;
    
    /**
     * @brief Prepares this matrix for being populated by `append`-calls.
     * 
     * See `append` for more details.
     */
    virtual void prepareAppend() = 0;
    
    /**
     * @brief Set the value at the given coordinates in this matrix, assuming
     * that nothing has been appended to coordinates after the given ones so
     * far.
     * 
     * Subsequent calls to this method must address strictly increasing
     * coordinates in the matrix, w.r.t. a row-major layout. E.g., a value may
     * be appended at (3, 2) after (3, 0) or (2, 3), but not after (3, 3) or
     * (4, 1).
     * 
     * Unlike `set`, this method does not assume that this matrix is in a
     * structurally valid state. However, `prepareAppend` must be called before
     * the first call to `append`. After a call to `append`, this matrix will
     * not necessarily be in a valid state. Thus, after a matrix has been
     * populated by `append`-calls, `finishAppend` must be called. To sum up,
     * the protocol is:
     * 
     * ```c++
     * Matrix * m = ...;
     * m->prepareAppend();
     * m->append(...); // as often as you want
     * m->finishAppend();
     * ```
     * 
     * Coordinates in this matrix not addressed by any `append`-call between
     * `prepareAppend` and `finishAppend` are assumed to be zero.
     * 
     * Note that populating a matrix by `append`-calls will overwrite any data
     * already present in the matrix.
     * 
     * You should refrain from mixing `set`, `append`, and direct access to the
     * underlying arrays to populate a matrix.
     * 
     * Expect this method to be inefficient and fall back to it only if not
     * avoidable. Use specific access methods of the particular sub-class
     * whenever possible.
     * 
     * @param rowIdx
     * @param colIdx
     * @param value
     */
    virtual void append(size_t rowIdx, size_t colIdx, ValueType value) = 0;
    
    /**
     * @brief Brings this matrix into a valid state after it has been populated
     * by `append`-calls.
     * 
     * See `append` for more details.
     */
    virtual void finishAppend() = 0;

    bool operator==(const Matrix<ValueType> & rhs) const {
        if (this == &rhs)
            return true;
        
        const size_t numRows = this->getNumRows();
        const size_t numCols = this->getNumCols();
        
        if (numRows != rhs.getNumRows() || numCols != rhs.getNumCols())
            return false;
        
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                if (this->get(r, c) != rhs.get(r, c))
                    return false;
        
        return true;
    }

    size_t getNumDims() const override {
        return 2;
    }

    size_t getNumItems() const override {
        return this->numCols * this->numRows;
    }
};

template<typename ValueType>
std::ostream & operator<<(std::ostream & os, const Matrix<ValueType> & obj) {
    obj.print(os);
    return os;
}

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_MATRIX_H
