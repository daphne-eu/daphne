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

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/MetaDataObject.h>

#include <cstddef>
#include <mutex>

/**
 * @brief The base class of all data structure implementations.
 */
class Structure
{
private:
    mutable size_t refCounter;
    mutable std::mutex refCounterMutex;
    
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);

protected:
    size_t row_offset{};
    size_t col_offset{};
    size_t numRows;
    size_t numCols;

    Structure(size_t numRows, size_t numCols);

    mutable std::shared_ptr<MetaDataObject> mdo;

    void clone_mdo(const Structure* src);
    
public:
    virtual ~Structure() = default;

    explicit operator std::unique_ptr<Range>() const {
        return std::make_unique<Range>(Range(0ul, 0ul, this->getNumRows(), this->getNumCols()));
    }

     explicit operator Range() const {
        return Range(0, 0, this->getNumRows(), this->getNumCols());
    }

    size_t getRefCounter() const {
        return refCounter;
    }
    
    MetaDataObject* getMetaDataObject() const {
        return mdo.get();
    }

    /**
     * @brief Increases the reference counter of this data object.
     * 
     * The access is protected by a mutex, such that multiple threads may call
     * this method concurrently.
     */
    void increaseRefCounter() const {
        refCounterMutex.lock();
        refCounter++;
        refCounterMutex.unlock();
    }
    
    // Note that there is no method for decreasing the reference counter here.
    // Instead, use DataObjectFactory::destroy(). It is important that the
    // reference counter becoming zero triggers the deletion of the data
    // object. Thus, we cannot handle it here.

    [[nodiscard]] virtual size_t getNumDims() const = 0;

    [[nodiscard]] size_t getNumRows() const
    {
        return numRows;
    }

    [[nodiscard]] size_t getNumCols() const
    {
        return numCols;
    }

    [[nodiscard]] virtual size_t getNumItems() const = 0;

    /**
     * @brief Prints a human-readable representation of this data object to the
     * given stream.
     * 
     * This method is not optimized for performance. It should only be used for
     * moderately small data objects.
     * 
     * @param os The stream where to print this data object.
     */
    virtual void print(std::ostream & os) const = 0;

    /**
     * @brief Extracts a row range out of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param rl Row range lower bound (inclusive).
     * @param ru Row range upper bound (exclusive).
     * @return 
     */
    virtual Structure* sliceRow(size_t rl, size_t ru) const = 0;

    /**
     * @brief Extracts a column range out of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param cl Column range lower bound (inclusive).
     * @param cu Column range upper bound (exclusive).
     * @return 
     */
    virtual Structure* sliceCol(size_t cl, size_t cu) const = 0;
    
    /**
     * @brief Extracts a rectangular sub-structure (row and column range) out
     * of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param rl Row range lower bound (inclusive).
     * @param ru Row range upper bound (exclusive).
     * @param cl Column range lower bound (inclusive).
     * @param cu Column range upper bound (exclusive).
     * @return 
     */
    virtual Structure* slice(size_t rl, size_t ru, size_t cl, size_t cu) const = 0;

    /**
     * @brief Serializes the object to a void buffer.     
     * 
     * @param buf buffer to store bytes.
     * @return The serialized buffer.
     */
    virtual size_t serialize(std::vector<char> &buf) const = 0;
};
