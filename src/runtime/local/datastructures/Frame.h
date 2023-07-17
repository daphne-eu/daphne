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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
class Frame : public Structure {
    
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
     * @brief An array of length `numCols` of the names of the columns of this
     * frame.
     */
    std::string * labels;
    
    /**
     * @brief A mapping from a column's label to its position in the frame.
     */
    std::unordered_map<std::string, size_t> labels2idxs;
    
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
    
    /**
     * @brief Initializes the mapping from column labels to column positions in
     * the frame and checks for duplicate column labels.
     * 
     * This method should be called by each constructor, after the column
     * labels have been initialized.
     */
    void initLabels2Idxs() {
        labels2idxs.clear();
        for(size_t i = 0; i < numCols; i++) {
            if(labels2idxs.count(labels[i]))
                throw std::runtime_error(
                        "a frame's column labels must be unique, but '" +
                        labels[i] + "' occurs more than once"
                );
            labels2idxs[labels[i]] = i;
        }
    }

    /**
     * @brief Initializes the mapping from column labels to column positions in
     * the frame and assigns default labels to duplicate column labels.
     * 
     * This method should only be called by constructors, that may intentionally duplicate 
     * columns, instead of initLabels2Idxs(), after the column labels have been initialized.
     */
    void initDeduplicatedLabels2Idxs() {
        labels2idxs.clear();
        for(size_t i = 0; i < numCols; i++) {
            if(labels2idxs.count(labels[i]))
               labels[i] = getDefaultLabel(i);
            labels2idxs[labels[i]] = i;
        }
    }
    
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
    Frame(size_t maxNumRows, size_t numCols, const ValueTypeCode * schema, const std::string * labels, bool zero) :
            Structure(maxNumRows, numCols),
            schema(new ValueTypeCode[numCols]),
            labels(new std::string[numCols]),
            columns(new std::shared_ptr<ColByteType>[numCols])
    {
        for(size_t i = 0; i < numCols; i++) {
            this->schema[i] = schema[i];
            this->labels[i] = labels ? labels[i] : getDefaultLabel(i);
            const size_t sizeAlloc = maxNumRows * ValueTypeUtils::sizeOf(schema[i]);
            this->columns[i] = std::shared_ptr<ColByteType>(new ColByteType[sizeAlloc],
                    std::default_delete<ColByteType []>());
            if(zero)
                memset(this->columns[i].get(), 0, sizeAlloc);
        }
        initLabels2Idxs();
    }
    
    Frame(const Frame * lhs, const Frame * rhs) :
            Structure(lhs->getNumRows(), lhs->getNumCols() + rhs->getNumCols())
    {
        if(lhs->getNumRows() != rhs->getNumRows())
            throw std::runtime_error(
                    "both input frames must have the same number of rows"
            );
        
        schema = new ValueTypeCode[numCols];
        labels = new std::string[numCols];
        columns = new std::shared_ptr<ColByteType>[numCols];
        
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numColsRhs = rhs->getNumCols();
        
        for(size_t i = 0; i < numColsLhs; i++) {
            schema [i] = lhs->schema[i];
            labels [i] = lhs->labels[i];
            columns[i] = std::shared_ptr<ColByteType>(lhs->columns[i]);
        }
        for(size_t i = 0; i < numColsRhs; i++) {
            schema [numColsLhs + i] = rhs->schema[i];
            labels [numColsLhs + i] = rhs->labels[i];
            columns[numColsLhs + i] = std::shared_ptr<ColByteType>(rhs->columns[i]);
        }
        initLabels2Idxs();
    }
    
    template<typename VT>
    bool tryValueType(Structure * colMat, ValueTypeCode * schemaSlot, std::shared_ptr<ColByteType> * columnsSlot) {
        if(auto colMat2 = dynamic_cast<DenseMatrix<VT> *>(colMat)) {
            assert(
                    (colMat2->getRowSkip() == 1) &&
                    "all given matrices must not be a view on a column of a larger matrix"
            );
            *schemaSlot = ValueTypeUtils::codeFor<VT>;
            std::shared_ptr<VT[]> orig = colMat2->getValuesSharedPtr();
            *columnsSlot = std::shared_ptr<ColByteType>(orig, reinterpret_cast<ColByteType *>(orig.get()));
            return true;
        }
        return false;
    }
    
    /**
     * @brief Creates a `Frame` with the given single-column matrices as its
     * columns.
     * 
     * The schema of the frame is automatically determined based on the value
     * types of the given matrices.
     * 
     * The data arrays are shared with the given matrices, i.e., no copying is
     * performed.
     * 
     * @param colMats A `std::vector` of single-column matrices. These must be
     * `DenseMatrix`s of any value type (the type `Structure` is used here only
     * to not depend on a template parameter for the value type). Furthermore,
     * these matrices must not be views on a single column of a larger matrix.
     */
    Frame(const std::vector<Structure *>& colMats, const std::string * labels) :
            Structure(colMats.empty() ? 0 : colMats[0]->getNumRows(), colMats.size())
    {
        const size_t numCols = colMats.size();
        assert(numCols && "you must provide at least one column matrix");
        schema = new ValueTypeCode[numCols];
        this->labels = new std::string[numCols];
        columns = new std::shared_ptr<ColByteType>[numCols];
        for(size_t c = 0; c < numCols; c++) {
            Structure * colMat = colMats[c];
            assert(
                    (colMat->getNumCols() == 1) &&
                    "all given matrices must have a single column"
            );
            assert(
                    (colMat->getNumRows() == numRows) &&
                    "all given column matrices must have the same number of rows"
            );
            this->labels[c] = labels ? labels[c] : getDefaultLabel(c);
            // For all value types.
            bool found = tryValueType<int8_t>(colMat, schema + c, columns + c);
            found = found || tryValueType<int32_t>(colMat, schema + c, columns + c);
            found = found || tryValueType<int64_t>(colMat, schema + c, columns + c);
            found = found || tryValueType<uint8_t> (colMat, schema + c, columns + c);
            found = found || tryValueType<uint32_t>(colMat, schema + c, columns + c);
            found = found || tryValueType<uint64_t>(colMat, schema + c, columns + c);
            found = found || tryValueType<float> (colMat, schema + c, columns + c);
            found = found || tryValueType<double>(colMat, schema + c, columns + c);
            if(!found)
                throw std::runtime_error("unsupported value type");
        }
        initLabels2Idxs();
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
    Frame(const Frame * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t numCols, const size_t * colIdxs) :
            Structure(rowUpperExcl - rowLowerIncl, numCols)
    {
        assert(src && "src must not be null");
        
        // Only check conditions, if input Frame has not zero rows and the expected output has not zero rows.
        if(!(rowLowerIncl == rowUpperExcl && rowLowerIncl == 0 && src->numRows == 0)) {
            assert(((rowLowerIncl < src->numRows) || rowLowerIncl == 0) && "rowLowerIncl is out of bounds");
            assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
            assert((rowLowerIncl <= rowUpperExcl) && "rowLowerIncl must be lower or equal than rowUpperExcl");
        }
        for(size_t i = 0; i < numCols; i++)
            assert((colIdxs[i] < src->numCols) && "some colIdx is out of bounds");
        
        this->schema = new ValueTypeCode[numCols];
        this->labels = new std::string[numCols];
        this->columns = new std::shared_ptr<ColByteType>[numCols];
        for(size_t i = 0; i < numCols; i++) {
            this->schema[i] = src->schema[colIdxs[i]];
            this->labels[i] = src->labels[colIdxs[i]];
            this->columns[i] = std::shared_ptr<ColByteType>(
                    src->columns[colIdxs[i]],
                    src->columns[colIdxs[i]].get() + rowLowerIncl * ValueTypeUtils::sizeOf(schema[i])
            );
        }
        initDeduplicatedLabels2Idxs();
    }
    
    ~Frame() override {
        delete[] schema;
        delete[] labels;
        delete[] columns;
    }
    
public:
    
    /**
     * @brief Returns the default label to use for the pos-th column, if no
     * column label was specified.
     * @param pos The position of the column in the frame (starting at zero).
     * @return The default label for the pos-th column.
     */
    static std::string getDefaultLabel(size_t pos) {
        return "col_" + std::to_string(pos);
    }
    
    void shrinkNumRows(size_t numRows) {
        // TODO Here we could reduce the allocated size of the column arrays.
        this->numRows = numRows;
    }
    
    const ValueTypeCode * getSchema() const {
        return schema;
    }
    
    const std::string * getLabels() const {
        return labels;
    }
    
    void setLabels(const std::string * newLabels) {
        for(size_t i = 0; i < numCols; i++)
            labels[i] = newLabels[i];
        initLabels2Idxs();
    }
    
    size_t getColumnIdx(const std::string & label) const {
        auto it = labels2idxs.find(label);
        if(it != labels2idxs.end())
            return it->second;
        throw std::runtime_error("column label not found: '" + label + "'");
    }
    
    ValueTypeCode getColumnType(size_t idx) const {
        assert((idx < numCols) && "column index is out of bounds");
        return schema[idx];
    }
    
    ValueTypeCode getColumnType(const std::string & label) const {
        return getColumnType(getColumnIdx(label));
    }
    
    template<typename ValueType>
    DenseMatrix<ValueType> * getColumn(size_t idx) {
        assert((ValueTypeUtils::codeFor<ValueType> == schema[idx]) && "requested value type must match the type of the column");
        return DataObjectFactory::create<DenseMatrix<ValueType>>(
                numRows, 1,
                std::shared_ptr<ValueType[]>(
                        columns[idx],
                        reinterpret_cast<ValueType *>(columns[idx].get())
                )
        );
    }
    
    template<typename ValueType>
    const DenseMatrix<ValueType> * getColumn(size_t idx) const {
        return const_cast<Frame *>(this)->getColumn<ValueType>(idx);
    }
    
    template<typename ValueType>
    DenseMatrix<ValueType> * getColumn(const std::string & label) {
        return getColumn<ValueType>(getColumnIdx(label));
    }
    
    template<typename ValueType>
    const DenseMatrix<ValueType> * getColumn(const std::string & label) const {
        return const_cast<Frame *>(this)->getColumn<ValueType>(label);
    }
    
    void * getColumnRaw(size_t idx) {
        return columns[idx].get();
    }
    
    const void * getColumnRaw(size_t idx) const {
        return const_cast<Frame *>(this)->getColumnRaw(idx);
    }
    
    void print(std::ostream & os) const override {
        os << "Frame(" << numRows << 'x' << numCols << ", [";
        for(size_t c = 0; c < numCols; c++) {
            // TODO Ideally, special characters in the labels should be
            // escaped.
            os << labels[c] << ':';
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

    Frame* sliceRow(size_t rl, size_t ru) const override {
        return slice(rl, ru, 0, numCols);
    }

    Frame* sliceCol(size_t cl, size_t cu) const override {
        return slice(0, numRows, cl, cu);
    }

    Frame* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        size_t colIdxs[cu-cl];
        size_t i = 0;
        for(size_t c = cl; c < cu; c++, i++)
            colIdxs[i] = c;
        return DataObjectFactory::create<Frame>(this, rl, ru, cu-cl, colIdxs);
    }
    size_t serialize(std::vector<char> &buf) const override;
};

std::ostream & operator<<(std::ostream & os, const Frame & obj);
