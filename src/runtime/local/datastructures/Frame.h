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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_FRAME_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_FRAME_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstring>

/**
 * @brief A data structure with an individual value type per column.
 * 
 * A `Frame` is organized in column-major fashion and is backed by an
 * individual dense array for each column.
 */
class Frame {
    size_t numRows;
    size_t numCols;
    
    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType * DataObjectFactory::create(ArgTypes ...);
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);
    
    /**
     * @brief An array of length `numCols` of the value types of the columns of
     * this frame.
     * 
     * Note that the schema is not encoded as template parameters since this
     * would lead to an explosion of frame types to be compiled.
     */
    ValueTypeCode * schema;
    
    /**
     * @brief The common pointer type used for the array of each column,
     * irrespective of the actual value type of the column.
     * 
     * Each column can have its own value type, as determined by the `schema`.
     * However, we cannot declare the column pointers of individual types,
     * since we want to store them in one array. Thus, we use a common pointer
     * type for all of them, internally.
     * 
     * Using `uint8_t` is advantageous, since `sizeof(uint8_t) == 1`, which
     * simplifies the computation of physical sizes.
     */
    using ColByteType = uint8_t;
    
    /**
     * @brief An array of length `numCols` of the column arrays of this frame.
     */
    std::shared_ptr<ColByteType> * columns;
    
    // TODO Should the given schema array really be copied, or reused?
    /**
     * @brief Creates a `Frame` and allocates enough memory for the specified
     * size.
     * 
     * @param maxNumRows The maximum number of rows.
     * @param numCols The exact number of columns.
     * @param schema An array of length `numCols` of the value types of the
     * individual columns. The given array will be copied.
     * @param zero Whether the allocated memory of the internal column arrays
     * shall be initialized to zeros (`true`), or be left uninitialized
     * (`false`).
     */
    Frame(size_t maxNumRows, size_t numCols, const ValueTypeCode * schema, bool zero) :
            numRows(maxNumRows),
            numCols(numCols),
            schema(new ValueTypeCode[numCols]),
            columns(new std::shared_ptr<ColByteType>[numCols])
    {
        for(size_t i = 0; i < numCols; i++) {
            this->schema[i] = schema[i];
            const size_t sizeAlloc = maxNumRows * ValueTypeUtils::sizeOf(schema[i]);
            this->columns[i] = std::shared_ptr<ColByteType>(new ColByteType[sizeAlloc]);
            if(zero)
                memset(this->columns[i].get(), 0, sizeAlloc);
        }
    }
    
    /**
     * @brief Creates a `Frame` around a sub-frame of another `Frame` without
     * copying the data.
     * 
     * @param src The other frame.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to extract.
     * @param rowUpperIncl Exclusive upper bound for the range of rows to extract.
     * @param numCols The number of columns to extract.
     * @param colIdxs An array of length `numCols` of the indexes of the
     * columns to extract from `src`.
     */
    Frame(const Frame * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t numCols, const size_t * colIdxs) {
        assert(src && "src must not be null");
        assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
        for(size_t i = 0; i < numCols; i++)
            assert((colIdxs[i] < src->numCols) && "some colIdx is out of bounds");
        
        this->numRows = rowUpperExcl - rowLowerIncl;
        this->numCols = numCols;
        this->schema = new ValueTypeCode[numCols];
        this->columns = new std::shared_ptr<ColByteType>[numCols];
        for(size_t i = 0; i < numCols; i++) {
            this->schema[i] = src->schema[colIdxs[i]];
            this->columns[i] = std::shared_ptr<ColByteType>(
                    src->columns[colIdxs[i]],
                    src->columns[colIdxs[i]].get() + rowLowerIncl * ValueTypeUtils::sizeOf(schema[i])
            );
        }
    }
    
    ~Frame() {
        delete[] columns;
        delete[] schema;
    }
    
public:
    
    size_t getNumRows() const {
        return numRows;
    }
    
    void shrinkNumRows(size_t numRows) {
        // TODO Here we could reduce the allocated size of the column arrays.
        this->numRows = numRows;
    }
    
    size_t getNumCols() const {
        return numCols;
    }
    
    const ValueTypeCode * getSchema() const {
        return schema;
    }
    
    ValueTypeCode getColumnType(size_t idx) const {
        assert((idx < numCols) && "column index is out of bounds");
        return schema[idx];
    }
    
    template<typename ValueType>
    DenseMatrix<ValueType> * getColumn(size_t idx) {
        assert((ValueTypeUtils::codeFor<ValueType> == schema[idx]) && "requested value type must match the type of the column");
        return DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, 1, reinterpret_cast<ValueType *>(columns[idx].get()));
    }
    
    template<typename ValueType>
    const DenseMatrix<ValueType> * getColumn(size_t idx) const {
        return const_cast<Frame *>(this)->getColumn<ValueType>(idx);
    }
    
    void print(std::ostream & os) const {
        os << "Frame(" << numRows << 'x' << numCols << ", [";
        for(size_t c = 0; c < numCols; c++) {
            os << ValueTypeUtils::cppNameForCode(schema[c]);
            if(c < numCols - 1)
                os << ", ";
        }
        os << "])" << std::endl;
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                ValueTypeUtils::printValue(os, schema[c], columns[c].get(), r);
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
    }
    
};

std::ostream & operator<<(std::ostream & os, const Frame & obj)
{
    obj.print(os);
    return os;
}

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_FRAME_H