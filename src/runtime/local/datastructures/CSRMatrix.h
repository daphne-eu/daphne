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
#include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <cstddef>
#include <cstring>
#include <type_traits>

/**
 * @brief A helper class to ensure that `CSRMatrix<ValueType>` (for general value types) has a `values` array, while
 * `CSRMatrix<bool>` (boolean value type) does not have a `values` array.
 *
 * This class stands between `CSRMatrix` and `Matrix` in the inheritance hierarchy. It has two template specializations,
 * one for `bool` and one for all non-`bool` value types.
 */
template <typename ValueType> class CSRMatrixStorage;

/**
 * @brief Template specialization of `CSRMatrixStorage` for all non-`bool` value types; ensures that the `values` array
 * exists.
 */
template <typename ValueType>
    requires(!std::same_as<ValueType, bool>)
class CSRMatrixStorage<ValueType> : public Matrix<ValueType> {
  protected:
    CSRMatrixStorage(size_t numRows, size_t numCols) : Matrix<ValueType>(numRows, numCols){};
    std::shared_ptr<ValueType[]> values; // values array
};

/**
 * @brief Template specialization of `CSRMatrixStorage` for the `bool` value type; ensures that the `values` array does
 * not exist.
 */
template <typename ValueType>
    requires(std::same_as<ValueType, bool>)
class CSRMatrixStorage<ValueType> : public Matrix<ValueType> {
  protected:
    CSRMatrixStorage(size_t numRows, size_t numCols) : Matrix<ValueType>(numRows, numCols){};
    // no values array
};

/**
 * @brief A sparse matrix in Compressed Sparse Row (CSR) format.
 *
 * This matrix implementation is backed by three contiguous arrays. The
 * `values` array contains all non-zero values in the matrix. For each of these
 * non-zero values, the `colIdxs` array contains the number of the column.
 * Finally, the `rowOffsets` array contains for each row in the matrix the
 * offset at which the corresponding entries can be found in the `values` and
 * `colIdxs` arrays. Additionally, the `rowOffsets` array ends with the offset
 * to the first element after the valid elements in the `values` and `colIdxs`
 * arrays.
 *
 * If the value type is `bool`, the non-zero values are not stored explicitly (since they would all be `true`) and the
 * `values` array as well as certain member functions and constructors directly related to the `values` array do not
 * exist.
 *
 * Each instance of this class might represent a sub-matrix of another
 * `CSRMatrix`. Thus, to traverse the matrix by row, you can safely go via the
 * `rowOffsets`, but for traversing the matrix by non-zero value, you must
 * start at `values[rowOffsets[0]]`.
 */
template <typename ValueType> class CSRMatrix : public CSRMatrixStorage<ValueType> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;

    /**
     * @brief The number of rows allocated starting from `rowOffsets`. This can
     * differ from `numRows` if this `CSRMatrix` is a view on a larger
     * `CSRMatrix`.
     */
    size_t numRowsAllocated;

    bool isRowAllocatedBefore;

    /**
     * @brief The maximum number of non-zero values this matrix was allocated
     * to accommodate.
     */
    size_t maxNumNonZeros;

    // Note that the values array is defined (or consciously not defined) in the superclass CSRMatrixStorage.
    std::shared_ptr<size_t[]> colIdxs;
    std::shared_ptr<size_t[]> rowOffsets;

    size_t lastAppendedRowIdx;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template <class DataType, typename... ArgTypes> friend DataType *DataObjectFactory::create(ArgTypes...);
    template <class DataType> friend void DataObjectFactory::destroy(const DataType *obj);

    /**
     * @brief Creates a `CSRMatrix` and allocates enough memory for the
     * specified size in the internal `values`, `colIdxs`, and `rowOffsets`
     * arrays.
     *
     * @param maxNumRows The maximum number of rows.
     * @param numCols The exact number of columns.
     * @param maxNumNonZeros The maximum number of non-zeros in the matrix.
     * @param zero Whether the allocated memory of the internal arrays shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    CSRMatrix(size_t maxNumRows, size_t numCols, size_t maxNumNonZeros, bool zero)
        : CSRMatrixStorage<ValueType>(maxNumRows, numCols), numRowsAllocated(maxNumRows), isRowAllocatedBefore(false),
          maxNumNonZeros(maxNumNonZeros), colIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()),
          rowOffsets(new size_t[numRows + 1], std::default_delete<size_t[]>()), lastAppendedRowIdx(0) {
        if constexpr (!std::is_same_v<ValueType, bool>)
            this->values =
                std::shared_ptr<ValueType[]>(new ValueType[maxNumNonZeros], std::default_delete<ValueType[]>());
        if (zero) {
            if constexpr (!std::is_same_v<ValueType, bool>)
                memset(this->values.get(), 0, maxNumNonZeros * sizeof(ValueType));
            memset(colIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
            memset(rowOffsets.get(), 0, (numRows + 1) * sizeof(size_t));
        }
    }

    /**
     * @brief Creates a `CSRMatrix` sharing the given values array and allocates enough memory for the specified size in
     * the `colIdxs` and `rowOffsets` arrays.
     *
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param numNonZeros The exact number of non-zeros.
     * @param values The values array to share. This should not be taken from a view into a larger `CSRMatrix`.
     */
    // TODO Support sharing the values array of a view into a CSRMatrix. This is a bit involved, since the matrix would
    // have a view for its values, but originals for its colIdxs and rowOffsets. So far, we don't support this case.
    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    CSRMatrix(size_t numRows, size_t numCols, size_t numNonZeros, std::shared_ptr<ValueType[]> &values)
        : CSRMatrixStorage<ValueType>(numRows, numCols), numRowsAllocated(numRows), isRowAllocatedBefore(false),
          maxNumNonZeros(numNonZeros), colIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()),
          rowOffsets(new size_t[numRows + 1], std::default_delete<size_t[]>()), lastAppendedRowIdx(0) {
        this->values = values;
    }

    /**
     * @brief Creates a `CSRMatrix` around a sub-matrix of another `CSRMatrix`
     * without copying the data.
     *
     * @param src The other `CSRMatrix`.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to
     * extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to
     * extract.
     */
    CSRMatrix(const CSRMatrix<ValueType> *src, size_t rowLowerIncl, size_t rowUpperExcl)
        : CSRMatrixStorage<ValueType>(rowUpperExcl - rowLowerIncl, src->numCols),
          numRowsAllocated(src->numRowsAllocated - rowLowerIncl), isRowAllocatedBefore(rowLowerIncl > 0),
          lastAppendedRowIdx(0) {
        if (!src)
            throw std::runtime_error("CSRMatrix: src must not be null");
        if (rowLowerIncl >= src->numRows)
            throw std::runtime_error("CSRMatrix: rowLowerIncl is out of bounds");
        if (rowUpperExcl > src->numRows)
            throw std::runtime_error("CSRMatrix: rowUpperExcl is out of bounds");
        if (rowLowerIncl >= rowUpperExcl)
            throw std::runtime_error("CSRMatrix: rowLowerIncl must be lower than rowUpperExcl");

        maxNumNonZeros = src->maxNumNonZeros;
        if constexpr (!std::is_same_v<ValueType, bool>)
            this->values = src->values;
        colIdxs = src->colIdxs;
        rowOffsets = std::shared_ptr<size_t[]>(src->rowOffsets, src->rowOffsets.get() + rowLowerIncl);
    }

    CSRMatrix(ValueType *v, size_t *cidx, size_t *ro, size_t nRows, size_t nCols, size_t maxNNZ,
              std::function<void(ValueType *)> valDeleter, std::function<void(size_t *)> idxDeleter)
        : CSRMatrixStorage<ValueType>(nRows, nCols), numRowsAllocated(nRows), isRowAllocatedBefore(false),
          maxNumNonZeros(maxNNZ), colIdxs(cidx, idxDeleter), rowOffsets(ro, idxDeleter),
          lastAppendedRowIdx(nRows ? nRows - 1 : 0) {
        if constexpr (!std::is_same_v<ValueType, bool>)
            this->values = std::shared_ptr<ValueType[]>(v, valDeleter);
    }

    virtual ~CSRMatrix() {
        // nothing to do
    }

    void fillNextPosUntil(size_t nextPos, size_t rowIdx) {
        if (rowIdx > lastAppendedRowIdx) {
            for (size_t r = lastAppendedRowIdx + 2; r <= rowIdx + 1; r++)
                rowOffsets.get()[r] = nextPos;
            lastAppendedRowIdx = rowIdx;
        }
    }

  public:
    template <typename NewValueType> using WithValueType = CSRMatrix<NewValueType>;

    static std::string getName() { return "CSRMatrix"; }

    void shrinkNumRows(size_t numRows) {
        if (numRows > this->numRows)
            throw std::runtime_error("CSRMatrix (shrinkNumRows): numRows can only be shrunk");
        // TODO Here we could reduce the allocated size of the rowOffsets array.
        this->numRows = numRows;
    }

    size_t getMaxNumNonZeros() const { return maxNumNonZeros; }
    size_t getNumNonZeros() const { return rowOffsets.get()[numRows] - rowOffsets.get()[0]; }

    size_t getNumNonZeros(size_t rowIdx) const {
        if (rowIdx >= numRows)
            throw std::runtime_error("CSRMatrix (getNumNonZeros): rowIdx is out of bounds");
        return rowOffsets.get()[rowIdx + 1] - rowOffsets.get()[rowIdx];
    }

    void shrinkNumNonZeros(size_t numNonZeros) {
        if (numNonZeros > getNumNonZeros())
            throw std::runtime_error("CSRMatrix (shrinkNumNonZeros): "
                                     "numNonZeros can only be shrunk");
        // TODO Here we could reduce the allocated size of the values and
        // colIdxs arrays.
    }

    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    ValueType *getValues() {
        return this->values.get();
    }

    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    const ValueType *getValues() const {
        return this->values.get();
    }

    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    std::shared_ptr<ValueType[]> getValuesSharedPtr() const {
        return this->values;
    }

    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    ValueType *getValues(size_t rowIdx) {
        // We allow equality here to enable retrieving a pointer to the end.
        if (rowIdx > numRows)
            throw std::runtime_error("CSRMatrix (getValues): rowIdx is out of bounds");
        return this->values.get() + rowOffsets.get()[rowIdx];
    }

    template <typename VT = ValueType>
        requires(!std::same_as<VT, bool>)
    const ValueType *getValues(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getValues(rowIdx);
    }

    size_t *getColIdxs() { return colIdxs.get(); }

    const size_t *getColIdxs() const { return colIdxs.get(); }

    std::shared_ptr<size_t[]> getColIdxsSharedPtr() const { return colIdxs; }

    size_t *getColIdxs(size_t rowIdx) {
        // We allow equality here to enable retrieving a pointer to the end.
        if (rowIdx > numRows)
            throw std::runtime_error("CSRMatrix (getColIdxs): rowIdx is out of bounds");
        return colIdxs.get() + rowOffsets.get()[rowIdx];
    }

    const size_t *getColIdxs(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getColIdxs(rowIdx);
    }

    size_t *getRowOffsets() { return rowOffsets.get(); }

    const size_t *getRowOffsets() const { return rowOffsets.get(); }

    std::shared_ptr<size_t[]> getRowOffsetsSharedPtr() const { return rowOffsets; }

    ValueType get(size_t rowIdx, size_t colIdx) const override {
        if (rowIdx >= numRows)
            throw std::runtime_error("CSRMatrix (get): rowIdx is out of bounds: " + std::to_string(rowIdx) +
                                     ", number of rows: " + std::to_string(numRows));
        if (colIdx >= numCols)
            throw std::runtime_error("CSRMatrix (get): colIdx is out of bounds: " + std::to_string(colIdx) +
                                     ", number of columns: " + std::to_string(numCols));

        const size_t *rowColIdxsBeg = getColIdxs(rowIdx);
        const size_t *rowColIdxsEnd = getColIdxs(rowIdx + 1);
        const size_t *ptrExpected = std::lower_bound(rowColIdxsBeg, rowColIdxsEnd, colIdx);

        if (ptrExpected == rowColIdxsEnd || *ptrExpected != colIdx) {
            // No entry for the given coordinates present.
            if constexpr (std::is_same_v<ValueType, bool>)
                return false;
            else
                return ValueType(0);
        } else {
            // Entry for the given coordinates present.
            if constexpr (std::is_same_v<ValueType, bool>)
                return true;
            else
                return getValues(rowIdx)[ptrExpected - rowColIdxsBeg];
        }
    }

    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        if (rowIdx >= numRows)
            throw std::runtime_error("CSRMatrix (set): rowIdx is out of bounds: " + std::to_string(rowIdx) +
                                     ", number of rows: " + std::to_string(numRows));
        if (colIdx >= numCols)
            throw std::runtime_error("CSRMatrix (set): colIdx is out of bounds: " + std::to_string(colIdx) +
                                     ", number of columns: " + std::to_string(numCols));

        size_t *rowColIdxsBeg = getColIdxs(rowIdx);
        size_t *rowColIdxsEnd = getColIdxs(rowIdx + 1);
        const size_t *ptrExpected = std::lower_bound(rowColIdxsBeg, rowColIdxsEnd, colIdx);
        const size_t posExpected = ptrExpected - rowColIdxsBeg;

        const size_t posEnd = colIdxs.get() + rowOffsets.get()[numRowsAllocated] - rowColIdxsBeg;

        if constexpr (std::is_same_v<ValueType, bool>) {
            if (ptrExpected == rowColIdxsEnd || *ptrExpected != colIdx) {
                // No entry for the given coordinates present.
                if (value == ValueType(0))
                    return; // do nothing
                else {
                    // Create gap.
                    // TODO We might want to reallocate here to ensure that enough
                    // space is allocated.
                    if (posEnd)
                        for (size_t pos = posEnd; pos > posExpected; pos--)
                            rowColIdxsBeg[pos] = rowColIdxsBeg[pos - 1];
                    // Insert given value and column index into the gap.
                    rowColIdxsBeg[posExpected] = colIdx;
                    // Update rowOffsets.
                    for (size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                        rowOffsets.get()[r]++;
                }
            } else {
                // Entry for the given coordinates present.
                if (value == ValueType(0)) {
                    // Close gap.
                    for (size_t pos = posExpected; pos < posEnd; pos++)
                        rowColIdxsBeg[pos] = rowColIdxsBeg[pos + 1];
                    // Update rowOffsets.
                    // TODO We might want to shrink the arrays here.
                    for (size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                        rowOffsets.get()[r]--;
                }
            }
        } else {
            ValueType *rowValuesBeg = getValues(rowIdx);

            if (ptrExpected == rowColIdxsEnd || *ptrExpected != colIdx) {
                // No entry for the given coordinates present.
                if (value == ValueType(0))
                    return; // do nothing
                else {
                    // Create gap.
                    // TODO We might want to reallocate here to ensure that enough
                    // space is allocated.
                    if (posEnd)
                        for (size_t pos = posEnd; pos > posExpected; pos--) {
                            rowValuesBeg[pos] = rowValuesBeg[pos - 1];
                            rowColIdxsBeg[pos] = rowColIdxsBeg[pos - 1];
                        }
                    // Insert given value and column index into the gap.
                    rowValuesBeg[posExpected] = value;
                    rowColIdxsBeg[posExpected] = colIdx;
                    // Update rowOffsets.
                    for (size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                        rowOffsets.get()[r]++;
                }
            } else {
                // Entry for the given coordinates present.
                if (value == ValueType(0)) {
                    // Close gap.
                    for (size_t pos = posExpected; pos < posEnd; pos++) {
                        rowValuesBeg[pos] = rowValuesBeg[pos + 1];
                        rowColIdxsBeg[pos] = rowColIdxsBeg[pos + 1];
                    }
                    // Update rowOffsets.
                    // TODO We might want to shrink the arrays here.
                    for (size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                        rowOffsets.get()[r]--;
                } else
                    // Simply overwrite the existing value.
                    rowValuesBeg[posExpected] = value;
            }
        }
    }

    void prepareAppend() override {
        if (isRowAllocatedBefore)
            // In this case, we assume that the matrix has been populated up to
            // just before this view.
            rowOffsets.get()[1] = rowOffsets.get()[0];
        else
            rowOffsets.get()[1] = rowOffsets.get()[0] = 0;
        lastAppendedRowIdx = 0;
    }

    // Note that if this matrix is a view on a larger `CSRMatrix`, then
    // `prepareAppend`/`append`/`finishAppend` assume that the larger matrix
    // has been populated up to just before the row range of this view.
    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        if (rowIdx >= numRows)
            throw std::runtime_error("CSRMatrix (append): rowIdx is out of bounds: " + std::to_string(rowIdx) +
                                     ", number of rows: " + std::to_string(numRows));
        if (colIdx >= numCols)
            throw std::runtime_error("CSRMatrix (append): colIdx is out of bounds: " + std::to_string(colIdx) +
                                     ", number of columns: " + std::to_string(numCols));

        if (value == ValueType(0))
            return;

        const size_t nextPos = rowOffsets.get()[lastAppendedRowIdx + 1];
        fillNextPosUntil(nextPos, rowIdx);

        if constexpr (!std::is_same_v<ValueType, bool>)
            this->values.get()[nextPos] = value;
        colIdxs.get()[nextPos] = colIdx;
        rowOffsets.get()[rowIdx + 1]++;
    }

    void finishAppend() override { fillNextPosUntil(rowOffsets.get()[lastAppendedRowIdx + 1], numRows - 1); }

    bool isView() const { return (numRowsAllocated > numRows || isRowAllocatedBefore); }

    void printValue(std::ostream &os, ValueType val) const {
        switch (ValueTypeUtils::codeFor<ValueType>) {
        case ValueTypeCode::SI8:
            os << static_cast<int32_t>(val);
            break;
        case ValueTypeCode::UI8:
            os << static_cast<uint32_t>(val);
            break;
        default:
            os << val;
            break;
        }
    }

    void print(std::ostream &os) const override {
        os << "CSRMatrix(" << numRows << 'x' << numCols << ", " << ValueTypeUtils::cppNameFor<ValueType> << ')'
           << std::endl;
        // Even though we generally assume that the colIdxs are sorted within each row, the following code could also
        // handle unsorted colIdxs (which is why it is a little complicated).
        ValueType *oneRow = new ValueType[numCols];
        for (size_t r = 0; r < numRows; r++) {
            memset(oneRow, 0, numCols * sizeof(ValueType));
            const size_t rowNumNonZeros = getNumNonZeros(r);
            const size_t *rowColIdxs = getColIdxs(r);
            if constexpr (std::is_same_v<ValueType, bool>) {
                for (size_t i = 0; i < rowNumNonZeros; i++)
                    oneRow[rowColIdxs[i]] = true;
            } else {
                const ValueType *rowValues = getValues(r);
                for (size_t i = 0; i < rowNumNonZeros; i++)
                    oneRow[rowColIdxs[i]] = rowValues[i];
            }
            for (size_t c = 0; c < numCols; c++) {
                printValue(os, oneRow[c]);
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
        delete[] oneRow;
    }

    /**
     * @brief Prints the internal arrays of this matrix.
     *
     * Meant to be used for testing and debugging only. Note that this method
     * works even if the internal state of the matrix is invalid, e.g., due to
     * uninitialized row offsets or column indexes.
     *
     * @param os The stream to print to.
     */
    void printRaw(std::ostream &os) const {
        os << "CSRMatrix(" << numRows << 'x' << numCols << ", " << ValueTypeUtils::cppNameFor<ValueType> << ')'
           << std::endl;
        os << "maxNumNonZeros: \t" << maxNumNonZeros << std::endl;
        if constexpr (!std::is_same_v<ValueType, bool>) {
            os << "values: \t";
            for (size_t i = 0; i < maxNumNonZeros; i++) {
                printValue(os, this->values.get()[i]);
                os << ", ";
            }
            os << std::endl;
        }
        os << "colIdxs: \t";
        for (size_t i = 0; i < maxNumNonZeros; i++)
            os << colIdxs.get()[i] << ", ";
        os << std::endl;
        os << "rowOffsets: \t";
        for (size_t i = 0; i <= numRows; i++)
            os << rowOffsets.get()[i] << ", ";
        os << std::endl;
    }

    CSRMatrix *sliceRow(size_t rl, size_t ru) const override {
        return DataObjectFactory::create<CSRMatrix>(this, rl, ru);
    }

    CSRMatrix *sliceCol(size_t cl, size_t cu) const override {
        // TODO add boundary validation when implementing
        throw std::runtime_error("CSRMatrix does not support sliceCol yet");
    }

    CSRMatrix *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        // TODO add boundary validation when implementing
        throw std::runtime_error("CSRMatrix does not support slice yet");
    }

    size_t bufferSize() { return this->getNumItems() * sizeof(ValueType); }

    bool operator==(const CSRMatrix<ValueType> &rhs) const {
        // Note that we do not use the generic `get` interface to matrices here
        // since this operator is meant to be used for writing tests for,
        // besides others, those generic interfaces.

        // Check if they are the same object.
        if (this == &rhs)
            return true;

        const size_t numRows = this->getNumRows();
        const size_t numCols = this->getNumCols();

        // Check if the shape matches.
        if (numRows != rhs.getNumRows() || numCols != rhs.getNumCols())
            return false;

        const size_t *rowOffsetsLhs = this->getRowOffsets();
        const size_t *rowOffsetsRhs = rhs.getRowOffsets();
        const size_t nnzLhs = rowOffsetsLhs[numRows] - rowOffsetsLhs[0];
        const size_t nnzRhs = rowOffsetsRhs[numRows] - rowOffsetsRhs[0];

        // Check if the total number of non-zeros matches.
        if (nnzLhs != nnzRhs)
            return false;

        // Check if the number of non-zeros matches for each row.
        // Both this and rhs could be a view (row segment) of a larger matrix. The row offsets of a view start with a
        // value greater than zero if and only if the original matrix has non-zeros in the rows before the ones covered
        // by the view. Thus, we can only compare the row offsets directly, if the row offsets arrays of both matrices
        // start with the same value. Otherwise, we subtract the difference of the offsets of row zero from all row
        // offsets of the matrix with the greater row offset of row zero. Alternatively, we could calculate the number
        // of non-zeros for each row as the delta of two subsequent row offsets and compare the number of non-zeros for
        // all rows (but that would require more calculations and memory accesses).
        if (rowOffsetsLhs[0] > rowOffsetsRhs[0]) {
            size_t diff = rowOffsetsLhs[0] - rowOffsetsRhs[0];
            for (size_t r = 1; r <= numRows; r++) // start at 1, because for 0 the check always succeeds due to diff
                if (rowOffsetsLhs[r] - diff != rowOffsetsRhs[r])
                    return false;
        } else if (rowOffsetsLhs[0] < rowOffsetsRhs[0]) {
            size_t diff = rowOffsetsRhs[0] - rowOffsetsLhs[0];
            for (size_t r = 1; r <= numRows; r++) // start at 1, because for 0 the check always succeeds due to diff
                if (rowOffsetsLhs[r] != rowOffsetsRhs[r] - diff)
                    return false;
        } else // rowOffsetsLhs[0] == rowOffsetsRhs[0]
            if (memcmp(rowOffsetsLhs, rowOffsetsRhs, (numRows + 1) * sizeof(size_t)))
                return false;

        // Check if the non-zero values match.
        if constexpr (!std::is_same_v<ValueType, bool>) {
            const ValueType *valuesLhs = this->getValues(0);
            const ValueType *valuesRhs = rhs.getValues(0);
            if (valuesLhs != valuesRhs)
                if (memcmp(valuesLhs, valuesRhs, nnzLhs * sizeof(ValueType)))
                    return false;
        }

        // Check if the column indices match.
        const size_t *colIdxsLhs = this->getColIdxs(0);
        const size_t *colIdxsRhs = rhs.getColIdxs(0);
        if (colIdxsLhs != colIdxsRhs)
            if (memcmp(colIdxsLhs, colIdxsRhs, nnzLhs * sizeof(size_t)))
                return false;

        return true;
    }

    size_t serialize(std::vector<char> &buf) const override;

    size_t getPhysicalSizeByte() const override {
        if constexpr (std::is_same_v<ValueType, bool>)
            return getNumNonZeros() * sizeof(size_t) // the size of the colIdxs array
                   + (numRows + 1) * sizeof(size_t); // the size of the rowOffsets array
        else
            return getNumNonZeros() * (sizeof(ValueType) + sizeof(size_t)) // the size of the values and colIdxs arrays
                   + (numRows + 1) * sizeof(size_t);                       // the size of the rowOffsets array
    }

    /**
     * @brief Checks if the underlying arrays of this `CSRMatrix` are populated in a valid way and throws an exception
     * with an explanation, otherwise.
     */
    void checkValidity() const {
        const size_t *rowOffsetsRaw = getRowOffsets();

        // Check the rowOffsets array.
        if (!isRowAllocatedBefore && rowOffsetsRaw[0] != 0)
            throw std::runtime_error("the first row offset must be zero unless this CSRMatrix is a view and there are "
                                     "rows allocated before it");
        for (size_t r = 1; r <= numRows; r++) {
            if (rowOffsetsRaw[r] < rowOffsetsRaw[r - 1])
                throw std::runtime_error("the row offsets must be an increasing sequence, violated at position " +
                                         std::to_string(r));
            const size_t nnzRow = rowOffsetsRaw[r] - rowOffsetsRaw[r - 1];
            if (nnzRow > numCols)
                throw std::runtime_error(
                    "a row may contain at most as many non-zeros as there are columns, violated in row " +
                    std::to_string(r - 1) + ", found " + std::to_string(nnzRow) + " but expected at most " +
                    std::to_string(numCols));
        }

        // Check the allocated size.
        const size_t nnz = rowOffsetsRaw[numRows];
        if (nnz > maxNumNonZeros)
            throw std::runtime_error("the rowOffsets array must not indicate more non-zeros than allocated, found " +
                                     std::to_string(nnz) + " but expected at most " + std::to_string(maxNumNonZeros));

        // Check the values and colIdxs arrays.
        for (size_t r = 0; r < numRows; r++) {
            const size_t nnzRow = getNumNonZeros(r);

            if constexpr (!std::is_same_v<ValueType, bool>) {
                // Check the values in the current row.
                const ValueType *valuesRow = getValues(r);
                for (size_t i = 0; i < nnzRow; i++)
                    if (valuesRow[i] == 0)
                        throw std::runtime_error("there must be no explicitly stored zeros, violated at position " +
                                                 std::to_string(i) + " in row " + std::to_string(r));
            }

            // Check the colIdxs in the current row.
            const size_t *colIdxsRow = getColIdxs(r);
            for (size_t i = 0; i < nnzRow; i++) {
                if (colIdxsRow[i] > numCols)
                    throw std::runtime_error("the column indexes must be in-bounds, violated at position " +
                                             std::to_string(i) + " in row " + std::to_string(r) + ", found " +
                                             std::to_string(colIdxsRow[i]) + " but expected at most " +
                                             std::to_string(numCols));
                if (i > 0 && colIdxsRow[i] <= colIdxsRow[i - 1])
                    throw std::runtime_error(
                        "the column indexes must be sorted and unique within each row, violated at position " +
                        std::to_string(i) + " in row " + std::to_string(r));
            }
        }
    }
};

template <typename ValueType> std::ostream &operator<<(std::ostream &os, const CSRMatrix<ValueType> &obj) {
    obj.print(os);
    return os;
}
