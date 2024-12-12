/*
 * Copyright 2024 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Helper functions
// ****************************************************************************

template <typename VTCol>
void innerJoinSetValue(DenseMatrix<VTCol> *res, const DenseMatrix<VTCol> *arg, const int64_t targetRow,
                       const int64_t fromRow, DCTX(ctx)) {
    const VTCol argValue = arg->get(fromRow, 0);
    res->set(targetRow, 0, argValue);
}

template <typename VTCol>
void innerJoinSet(ValueTypeCode vtcType, Frame *&res, const Frame *arg, const int64_t toRow, const int64_t toCol,
                  const int64_t fromRow, const int64_t fromCol, DCTX(ctx)) {
    if (vtcType == ValueTypeUtils::codeFor<VTCol>) {
        const DenseMatrix<VTCol> *colArg = arg->getColumn<VTCol>(fromCol);
        DenseMatrix<VTCol> *colRes = res->getColumn<VTCol>(toCol);
        innerJoinSetValue<VTCol>(colRes, colArg, toRow, fromRow, ctx);
        DataObjectFactory::destroy(colArg, colRes);
    }
}

// Create a hash table for rhs
template <typename VTRhs>
std::unordered_map<VTRhs, std::vector<size_t>> BuildHashRhs(const Frame *rhs, const char *rhsOn,
                                                            const size_t numRowRhs) {
    std::unordered_map<VTRhs, std::vector<size_t>> res;
    const DenseMatrix<VTRhs> *col = rhs->getColumn<VTRhs>(rhsOn);
    for (size_t row_idx_r = 0; row_idx_r < numRowRhs; row_idx_r++) {
        VTRhs key = col->get(row_idx_r, 0);
        res[key].push_back(row_idx_r);
    }
    DataObjectFactory::destroy(col);
    return res;
}

template <typename VT>
int64_t ProbeHashLhs(
    // results and results schema
    Frame *&res, ValueTypeCode *schema,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn,
    // num columns
    const size_t numColRhs, const size_t numColLhs,
    // context
    DCTX(ctx),
    // hashed map of Rhs
    std::unordered_map<VT, std::vector<size_t>> hashRhsIndex,
    // Lhs rowa
    const size_t numRowLhs) {
    int64_t row_idx_res = 0;
    int64_t col_idx_res = 0;
    auto lhsFKCol = lhs->getColumn<VT>(lhsOn);
    for (size_t row_idx_l = 0; row_idx_l < numRowLhs; row_idx_l++) {
        auto key = lhsFKCol->get(row_idx_l, 0);
        auto it = hashRhsIndex.find(key);

        if (it != hashRhsIndex.end()) {
            for (size_t row_idx_r : it->second) {
                col_idx_res = 0;

                // Populate result row from lhs columns
                for (size_t idx_c = 0; idx_c < numColLhs; idx_c++) {
                    innerJoinSet<std::string>(schema[col_idx_res], res, lhs, row_idx_res, col_idx_res, row_idx_l, idx_c,
                                              ctx);
                    innerJoinSet<int64_t>(schema[col_idx_res], res, lhs, row_idx_res, col_idx_res, row_idx_l, idx_c,
                                          ctx);
                    innerJoinSet<double>(schema[col_idx_res], res, lhs, row_idx_res, col_idx_res, row_idx_l, idx_c,
                                         ctx);

                    col_idx_res++;
                }

                // Populate result row from rhs columns
                for (size_t idx_c = 0; idx_c < numColRhs; idx_c++) {
                    innerJoinSet<std::string>(schema[col_idx_res], res, rhs, row_idx_res, col_idx_res, row_idx_r, idx_c,
                                              ctx);
                    innerJoinSet<int64_t>(schema[col_idx_res], res, rhs, row_idx_res, col_idx_res, row_idx_r, idx_c,
                                          ctx);

                    innerJoinSet<double>(schema[col_idx_res], res, rhs, row_idx_res, col_idx_res, row_idx_r, idx_c,
                                         ctx);

                    col_idx_res++;
                }

                row_idx_res++;
            }
        }
    }
    DataObjectFactory::destroy(lhsFKCol);
    return row_idx_res;
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

inline void innerJoin(
    // results
    Frame *&res,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn, const char *rhsOn,
    // result size
    int64_t numRowRes,
    // context
    DCTX(ctx)) {
    // Find out the value types of the columns to process.
    ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);

    // Perhaps check if res already allocated.
    const size_t numRowRhs = rhs->getNumRows();
    const size_t numRowLhs = lhs->getNumRows();
    const size_t totalRows = numRowRes == -1 ? numRowRhs * numRowLhs : numRowRes;
    const size_t numColRhs = rhs->getNumCols();
    const size_t numColLhs = lhs->getNumCols();
    const size_t totalCols = numColRhs + numColLhs;
    const std::string *oldlabels_l = lhs->getLabels();
    const std::string *oldlabels_r = rhs->getLabels();

    int64_t col_idx_res = 0;
    int64_t row_idx_res;

    // Set up schema and labels
    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    for (size_t col_idx_l = 0; col_idx_l < numColLhs; col_idx_l++) {
        schema[col_idx_res] = lhs->getColumnType(col_idx_l);
        newlabels[col_idx_res++] = oldlabels_l[col_idx_l];
    }
    for (size_t col_idx_r = 0; col_idx_r < numColRhs; col_idx_r++) {
        schema[col_idx_res] = rhs->getColumnType(col_idx_r);
        newlabels[col_idx_res++] = oldlabels_r[col_idx_r];
    }

    // Initialize result frame with an estimate
    res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, newlabels, false);

    // Build hash table and prob left table
    if (vtcLhsOn == ValueTypeCode::STR) {
        row_idx_res = ProbeHashLhs<std::string>(res, schema, lhs, rhs, lhsOn, numColRhs, numColLhs, ctx,
                                                BuildHashRhs<std::string>(rhs, rhsOn, numRowRhs), numRowLhs);
    } else {
        row_idx_res = ProbeHashLhs<int64_t>(res, schema, lhs, rhs, lhsOn, numColRhs, numColLhs, ctx,
                                            BuildHashRhs<int64_t>(rhs, rhsOn, numRowRhs), numRowLhs);
    }
    // Shrink result frame to actual size
    res->shrinkNumRows(row_idx_res);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
