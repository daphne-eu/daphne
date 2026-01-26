/*
 * Copyright 2026 The DAPHNE Consortium
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

#include <cstddef>
#include <runtime/local/datastructures/CSRMatrix.h>

/**
 * @brief Lightweight struct to compute and hold sparse matrix statistics.
 *
 * All operations are O(1) since getNumNonZeros() uses row offset arithmetic.
 */
struct CSRStats {
    size_t nnz;
    size_t numRows;
    size_t numCols;
    double sparsity; // nnz / (rows * cols), range [0.0, 1.0]

    template <typename VT> static CSRStats compute(const CSRMatrix<VT> *mat) {
        CSRStats stats;
        stats.numRows = mat->getNumRows();
        stats.numCols = mat->getNumCols();
        stats.nnz = mat->getNumNonZeros();
        stats.sparsity = static_cast<double>(stats.nnz) / (static_cast<double>(stats.numRows) * stats.numCols);
        return stats;
    }
};
