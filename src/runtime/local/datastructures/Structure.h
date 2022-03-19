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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_STRUCTURE_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_STRUCTURE_H

#include <iostream>

#include <cstddef>

/**
 * @brief The base class of all data structure implementations.
 */
class Structure
{
protected:
    size_t numRows;
    size_t numCols;

    Structure(size_t numRows, size_t numCols) :
            numRows(numRows), numCols(numCols)
    {
        // nothing to do
    };

public:
    virtual ~Structure() = default;

    [[nodiscard]] size_t getNumRows() const
    {
        return numRows;
    }

    [[nodiscard]] size_t getNumCols() const
    {
        return numCols;
    }

    [[nodiscard]] size_t getNumItems() const
    {
        return numRows * numCols;
    }

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
};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_STRUCTURE_H