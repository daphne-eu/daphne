/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <ir/daphneir/Daphne.h>
#include <runtime/local/kernels/Transpose.h>
#include <util/preprocessor_defs.h>

#include <mutex>
#include <queue>

using mlir::daphne::VectorCombine;

template <typename DT> class VectorizedDataSink {
  public:
    void add(DT *matrix, size_t startRow) = delete;
};

// TODO: VectorizedDataSink for DenseMatrix

template <typename VT> class VectorizedDataSink<CSRMatrix<VT>> {
    using QueueElements = std::pair<size_t, CSRMatrix<VT> *>;
    VectorCombine _combine;
    std::priority_queue<QueueElements, std::vector<QueueElements>,
                        std::greater<>>
        _results;
    std::mutex _mtx;
    uint64_t _numRows = 0;
    uint64_t _numCols = 0;
    uint64_t _numNnz = 0;
    // for column-wise combine
    std::vector<size_t> _rowNnz;

  public:
    VectorizedDataSink(VectorCombine combine, uint64_t numRows,
                       uint64_t numCols)
        : _combine(combine), _numRows(numRows), _numCols(numCols),
          _rowNnz(numRows) {}

    void add(CSRMatrix<VT> *matrix, uint64_t startRow,
             bool multiThreaded = true) {
        std::unique_lock<std::mutex> lock(_mtx, std::defer_lock);
        if (multiThreaded) {
            lock.lock();
        }
        switch (_combine) {
        case VectorCombine::COLS: {
            auto rows = matrix->getNumRows();
            auto *off = matrix->getRowOffsets();
            auto *rowNnz = &_rowNnz[0];
            PRAGMA_LOOP_VECTORIZE
            for (size_t row = 0; row < rows; ++row) {
                rowNnz[row] += off[row + 1] - off[row];
            }
        }
            LLVM_FALLTHROUGH;
        case VectorCombine::ROWS: {
            _numNnz += matrix->getNumNonZeros();
            _results.emplace(startRow, matrix);
            break;
        }
        default: {
            throw std::runtime_error("Vectorization of sparse matrices only "
                                     "implemented for row-wise combines");
        }
        }
    }

    CSRMatrix<VT> *consume() {
        if (_results.empty()) {
            throw std::runtime_error(
                "Vectorized CSRMatrix without any iterations");
        }
        auto *res = DataObjectFactory::create<CSRMatrix<VT>>(_numRows, _numCols,
                                                             _numNnz, false);
        auto *resRowOff = res->getRowOffsets();
        resRowOff[0] = 0;
        auto *resValues = res->getValues();
        auto *resColIdxs = res->getColIdxs();
        if (_combine == VectorCombine::ROWS) {
            while (!_results.empty()) {
                auto pair = _results.top();
                _results.pop();

                auto rowStart = pair.first;
                auto currMat = pair.second;
                auto *currRowOff = currMat->getRowOffsets();
                for (size_t row = 0; row < currMat->getNumRows(); ++row) {
                    resRowOff[rowStart + row + 1] =
                        resRowOff[rowStart] + currRowOff[row + 1];
                }
                std::memcpy(resColIdxs + resRowOff[rowStart],
                            currMat->getColIdxs(),
                            currMat->getNumNonZeros() * sizeof(*resColIdxs));
                std::memcpy(resValues + resRowOff[rowStart],
                            currMat->getValues(),
                            currMat->getNumNonZeros() * sizeof(*resValues));
                DataObjectFactory::destroy(currMat);
            }
        } else if (_combine == VectorCombine::COLS) {
            auto *rowNnz = &_rowNnz[0];
            PRAGMA_LOOP_VECTORIZE
            for (size_t row = 0; row < _numRows; ++row) {
                resRowOff[row + 1] = resRowOff[row] + rowNnz[row];
                // reuse rowNnz to place new values
                rowNnz[row] = resRowOff[row];
            }
            // we start with first columns
            while (!_results.empty()) {
                auto pair = _results.top();
                _results.pop();

                auto colStart = pair.first;
                auto currMat = pair.second;
                auto *currRowOff = currMat->getRowOffsets();
                auto *currValues = currMat->getValues();
                auto *currColIdxs = currMat->getColIdxs();
                for (size_t row = 0; row < currMat->getNumRows(); ++row) {
                    auto offset = currRowOff[row];
                    auto len = currRowOff[row + 1] - offset;
                    std::memcpy(resValues + rowNnz[row], currValues + offset,
                                len * sizeof(*resValues));
                    for (size_t i = 0; i < len; ++i) {
                        resColIdxs[rowNnz[row] + i] =
                            colStart + currColIdxs[offset + i];
                    }
                    rowNnz[row] += len;
                }
                DataObjectFactory::destroy(currMat);
            }
        } else {
            llvm_unreachable("NOT IMPLEMENTED");
        }

        return res;
    }
};
