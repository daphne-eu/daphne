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
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <iomanip>

#include <cassert>
#include <cstddef>
#include <cstring>

/**
 * @brief A sparse matrix in COOrdinate (COO) format.
 *
 */
template<typename ValueType>
class COOMatrix : public Matrix<ValueType> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;

    /**
     * @brief The maximum number of non-zero values this matrix was allocated
     * to accommodate.
     */
    size_t maxNumNonZeros;

    /**
     * @brief We need these in order to accommodate row-based sub-matrix views.
     */
    size_t lowerRow;
    size_t upperRow;

    size_t appendHelp = 0;
    bool view;

    std::shared_ptr<ValueType> values;
    std::shared_ptr<size_t> colIdxs;
    std::shared_ptr<size_t> rowIdxs;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType *DataObjectFactory::create(ArgTypes ...);

    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType *obj);

    /**
     * @brief Creates a `COOMatrix` and allocates enough memory for the
     * specified size in the internal `values`, `colIdxs`, and `rowIdxs`
     * arrays.
     *
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param maxNumNonZeros The maximum number of non-zeros in the matrix.
     * @param zero Whether the allocated memory of the internal arrays shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    COOMatrix(size_t numRows, size_t numCols, size_t maxNumNonZeros, bool zero) :
            Matrix<ValueType>(numRows, numCols),
            maxNumNonZeros(maxNumNonZeros),
            lowerRow(0),
            upperRow(numRows),
            view(false),
            values(new ValueType[maxNumNonZeros + 1], std::default_delete<ValueType[]>()),
            colIdxs(new size_t[maxNumNonZeros + 1], std::default_delete<size_t[]>()),
            rowIdxs(new size_t[maxNumNonZeros + 1], std::default_delete<size_t[]>()) {
        if (zero) {
            memset(values.get(), 0, (maxNumNonZeros + 1) * sizeof(ValueType));
            memset(colIdxs.get(), 0, (maxNumNonZeros + 1) * sizeof(size_t));
            memset(rowIdxs.get(), 0, (maxNumNonZeros + 1) * sizeof(size_t));
        }

        values.get()[0] = ValueType(0);
        rowIdxs.get()[0] = size_t(-1);
        colIdxs.get()[0] = size_t(-1);
    }

    /**
     * @brief Creates a `COOMatrix` around a sub-matrix of another `COOMatrix`
     * without copying the data.
     *
     * @param src The other `COOMatrix`.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to extract.
     */
    COOMatrix(const COOMatrix<ValueType> *src, size_t rowLowerIncl, size_t rowUpperExcl) :
            Matrix<ValueType>(rowUpperExcl - rowLowerIncl, src->numCols),
            maxNumNonZeros(std::min(src->maxNumNonZeros, src->numCols * (rowUpperExcl - rowLowerIncl))),
            lowerRow(rowLowerIncl),
            upperRow(rowUpperExcl),
            view(true) {
        assert(src && "src must not be null");
        assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");

        rowIdxs = src->rowIdxs;
        colIdxs = src->colIdxs;
        values = src->values;
    }

    virtual ~COOMatrix() {
        // nothing to do
    }

    [[nodiscard]] std::pair<size_t, size_t> rowRange(size_t rowIdx, size_t start) const {
        size_t rowStart = 0, rowLength = 0, row, i = start;
        while (true) {
            row = rowIdxs.get()[i];
            if (row == (size_t) -1) break;
            if (row < rowIdx) {
                i++;
                continue;
            }
            if (row > rowIdx) {
                if (rowLength == 0) rowStart = i;
                return std::make_pair(rowStart, rowLength);
            }
            if (row == rowIdx) {
                if (rowLength == 0) rowStart = i;
                rowLength++;
            }
            i++;
        }

        if (rowLength == 0) return std::make_pair(i, rowLength);
        return std::make_pair(rowStart, rowLength);
    }

    void insert(size_t pos, size_t rowIdx, size_t colIdx, ValueType value) {
        assert((this->getNumNonZeros() < maxNumNonZeros) && "can't add any more nonzero values");

        if (value == ValueType(0)) return;
        ValueType val = values.get()[pos];
        size_t row = rowIdxs.get()[pos];
        size_t col = colIdxs.get()[pos];
        size_t i = pos + 1;
        while (true) {
            if (row == size_t(-1)) {
                std::swap(val, values.get()[i]);
                std::swap(row, rowIdxs.get()[i]);
                std::swap(col, colIdxs.get()[i]);
                break;
            }
            std::swap(val, values.get()[i]);
            std::swap(row, rowIdxs.get()[i]);
            std::swap(col, colIdxs.get()[i]);

            i++;
        }
        rowIdxs.get()[pos] = rowIdx;
        values.get()[pos] = value;
        colIdxs.get()[pos] = colIdx;
    }

    void remove(size_t idx) {
        size_t i = idx;
        while (true) {
            rowIdxs.get()[i] = rowIdxs.get()[i + 1];
            values.get()[i] = values.get()[i + 1];
            colIdxs.get()[i] = colIdxs.get()[i + 1];
            if (rowIdxs.get()[i] == size_t(-1)) break;
            i++;
        }
    }

public:
    [[nodiscard]] bool isView() const {
        return view;
    }

    [[nodiscard]] size_t getLowerRow() const {
        return lowerRow;
    }

    [[nodiscard]] size_t getUpperRow() const {
        return upperRow;
    }

    [[nodiscard]] size_t getMaxNumNonZeros() const {
        return maxNumNonZeros;
    }

    [[nodiscard]] size_t getNumNonZeros() const {
        size_t i = 0, cnt = 0;
        while (true) {
            size_t row = rowIdxs.get()[i];
            if (row == size_t(-1)) break;
            if (row < lowerRow) {
                i++;
                continue;
            }
            if (row >= upperRow) break;
            i++;
            cnt++;
        }
        return cnt;
    }

    [[nodiscard]] size_t getNumNonZerosRow(size_t rowIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx + lowerRow, 0);
        return range.second;
    }

    [[nodiscard]] size_t getNumNonZerosCol(size_t colIdx) const {
        assert((colIdx < numCols) && "colIdx is out of bounds");

        size_t cnt = 0, i = 0;
        while (true) {
            size_t col = colIdxs.get()[i];
            size_t row = rowIdxs.get()[i];
            if (col == size_t(-1)) break;
            if (col == colIdx && row >= lowerRow && row < upperRow) cnt++;
            i++;
        }
        return cnt;
    }

    [[nodiscard]] ValueType *getValues() {
        std::pair<size_t, size_t> range = rowRange(lowerRow, 0);
        size_t rowStart = range.first;

        return values.get() + rowStart;
    }

    [[nodiscard]] const ValueType *getValues() const {
        return const_cast<COOMatrix<ValueType> *>(this)->getValues();
    }

    [[nodiscard]] size_t *getColIdxs() {
        std::pair<size_t, size_t> range = rowRange(lowerRow, 0);
        size_t rowStart = range.first;

        return colIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getColIdxs() const {
        return const_cast<COOMatrix<ValueType> *>(this)->getColIdxs();
    }

    [[nodiscard]] size_t *getRowIdxs() {
        std::pair<size_t, size_t> range = rowRange(lowerRow, 0);
        size_t rowStart = range.first;

        return rowIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getRowIdxs() const {
        return const_cast<COOMatrix<ValueType> *>(this)->getRowIdxs();
    }

    [[nodiscard]] ValueType *getValues(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx + lowerRow, 0);
        size_t rowStart = range.first;

        return values.get() + rowStart;
    }

    [[nodiscard]] const ValueType *getValues(size_t rowIdx) const {
        return const_cast<COOMatrix<ValueType> *>(this)->getValues(rowIdx);
    }

    [[nodiscard]] size_t *getColIdxs(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx + lowerRow, 0);
        size_t rowStart = range.first;

        return colIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getColIdxs(size_t rowIdx) const {
        return const_cast<COOMatrix<ValueType> *>(this)->getColIdxs(rowIdx);
    }

    [[nodiscard]] size_t *getRowIdxs(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx + lowerRow, 0);
        size_t rowStart = range.first;

        return rowIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getRowIdxs(size_t rowIdx) const {
        return const_cast<COOMatrix<ValueType> *>(this)->getRowIdxs(rowIdx);
    }

    [[nodiscard]] ValueType get(size_t rowIdx, size_t colIdx) const override {
        rowIdx += lowerRow;

        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");

        for (size_t i = 0; i < maxNumNonZeros; i++) {
            size_t row = rowIdxs.get()[i];
            if (row == size_t(-1)) break;
            if (row > rowIdx) break;
            if (row < rowIdx) continue;
            size_t col = colIdxs.get()[i];
            if (col > colIdx) break;
            if (col < colIdx) continue;
            if (col == colIdx) return values.get()[i];
        }
        return ValueType(0);
    }

    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        rowIdx += lowerRow;

        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");

        for (size_t i = 0; i < maxNumNonZeros; i++) {
            size_t row = rowIdxs.get()[i];
            if (row == size_t(-1)) {
                insert(i, rowIdx, colIdx, value);
                return;
            }
            if (row > rowIdx) {
                insert(i, rowIdx, colIdx, value);
                return;
            }
            if (row < rowIdx) continue;
            size_t col = colIdxs.get()[i];
            if (col > colIdx) {
                insert(i, rowIdx, colIdx, value);
                return;
            }
            if (col < colIdx) continue;
            if (col == colIdx) {
                if (value == ValueType(0)) {
                    remove(i);
                    return;
                } else {
                    values.get()[i] = value;
                    return;
                }
            }
        }
    }

    void prepareAppend() override {
        appendHelp = rowRange(lowerRow, 0).first;
    }

    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        assert((appendHelp < maxNumNonZeros) && "can't add any more nonzero values");

        if (value == ValueType(0)) return;

        rowIdxs.get()[appendHelp] = rowIdx;
        values.get()[appendHelp] = value;
        colIdxs.get()[appendHelp] = colIdx;
        appendHelp++;
    }

    void finishAppend() override {
        rowIdxs.get()[appendHelp] = size_t(-1);
        values.get()[appendHelp] = ValueType(0);
        colIdxs.get()[appendHelp] = size_t(-1);
    }

    /**
     * @brief Pretty print of this matrix.
     * @param os The stream to print to.
     */
    void print(std::ostream &os) const override {
        os << "COOMatrix(" << numRows << 'x' << numCols << ", "
           << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl << std::endl;

        auto *colWidths = new int[numCols];
        for (size_t i = 0; i < numCols; ++i) {
            colWidths[i] = 1;
        }

        size_t index = 0;
        while (true) {
            ValueType value = values.get()[index];
            if (value == ValueType(0)) break;
            std::ostringstream oss;
            oss << value;
            std::string strValue = oss.str();
            colWidths[colIdxs.get()[index]] = std::max(static_cast<int>(strValue.length()),
                                                       colWidths[colIdxs.get()[index]]);
            index++;
        }

        size_t i = 0;
        ValueType val = values.get()[i];
        size_t row = rowIdxs.get()[i];
        size_t col = colIdxs.get()[i];
        for (size_t currRow = 0; currRow < numRows; ++currRow) {
            for (size_t currCol = 0; currCol < numCols; ++currCol) {
                if (currRow == row && currCol == col) {
                    os << std::setw(colWidths[col]) << val << " ";
                    i++;
                    val = values.get()[i];
                    row = rowIdxs.get()[i];
                    col = colIdxs.get()[i];
                } else {
                    os << std::setw(colWidths[col]) << ValueType(0) << " ";
                }
            }
            os << std::endl;
        }

        delete[] colWidths;
    }

    /**
     * @brief Prints the internal arrays of this matrix.
     * @param os The stream to print to.
     */
    void printRaw(std::ostream &os) const {
        size_t numNonZeros = this->getNumNonZeros();
        os << "COOMatrix(" << numRows << 'x' << numCols << ", "
           << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;
        os << "maxNumNonZeros: \t" << maxNumNonZeros << std::endl;
        os << "numNonZeros: \t" << this->getNumNonZeros() << std::endl;
        os << "values: \t";
        size_t offset = rowRange(lowerRow, 0).first;
        for (size_t i = 0; i < numNonZeros; i++)
            os << values.get()[i + offset] << ", ";
        os << std::endl;
        os << "colIdxs: \t";
        for (size_t i = 0; i < numNonZeros; i++)
            os << colIdxs.get()[i + offset] << ", ";
        os << std::endl;
        os << "rowIdxs: \t";
        for (size_t i = 0; i < numNonZeros; i++)
            os << rowIdxs.get()[i + offset] << ", ";
        os << std::endl;
    }

    COOMatrix *sliceRow(size_t rl, size_t ru) const override {
        return DataObjectFactory::create<COOMatrix>(this, rl, ru);
    }

    COOMatrix *sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("COOMatrix does not support column-based slicing yet");
    }

    COOMatrix *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("COOMatrix does not support slicing yet");
    }

    [[nodiscard]] size_t bufferSize() {
        return (maxNumNonZeros + 1) * (sizeof(ValueType) + sizeof(size_t) + sizeof(size_t));
    }

    size_t serialize(std::vector<char> &buf) const override;
};

template<typename ValueType>
std::ostream &operator<<(std::ostream &os, const COOMatrix<ValueType> &obj) {
    obj.print(os);
    return os;
}