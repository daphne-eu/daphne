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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/ExtractRow.h>
#include <util/DeduceType.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Order {
    static void apply(DT *& res, const DT * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

// Note that we normally don't pass any arguments after the DaphneContext. In
// this case it is only okay because groupsRes has a default and is meant to be
// used only by other kernels, not from DaphneDSL.
template<class DT>
void order(DT *& res, const DT * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
    Order<DT>::apply(res, arg, colIdxs, numColIdxs, ascending, numAscending, returnIdx, ctx, groupsRes);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

// scans a column for groups of duplicates (stored as VTIdx pairs forming index ranges)
// performed only within the index ranges in the input groups vector on the values of the input column (DenseMatrix)
// replaces the groups in the input vector with the groups found during the scan
template<typename VTIdx, typename VT>
void columnGroupScan(std::vector<std::pair<VTIdx, VTIdx>> &groups, DenseMatrix<VT>* col, DCTX(ctx)) {
    const size_t numOldGroups = groups.size();
    const VT * values = col->getValues();
    for (size_t g = 0; g < numOldGroups; g++) {
        auto first = values + groups[g].first;
        const auto last = values + groups[g].second;
        while (first != last) {
            const auto next{std::find_if(first + 1, last, [&](const VT& t) {return t != *first;})};
            if (next - first > 1) {
                groups.push_back(std::make_pair(first-values, next-values));
            }
            first = next;
        }
    }
    groups.erase(groups.begin(),groups.begin()+numOldGroups);
}

// sorts input idx DenseMatrix within the index ranges in the input groups vector on the values of the input column (DenseMatrix)
template<typename VTIdx, typename VT>
void columnIDSort(DenseMatrix<VTIdx> *&idx, const DenseMatrix<VT>* col, std::vector<std::pair<VTIdx, VTIdx>> &groups, bool ascending, DCTX(ctx)) {
    VTIdx * indicies = idx->getValues();
    const VT * values = col->getValues();
    auto compare = ascending ? std::function{[&values](VTIdx i, VTIdx j) {return values[i] < values[j];}}
                             : std::function{[&values](VTIdx i, VTIdx j) {return values[i] > values[j];}};
    for (const auto &group : groups)
        std::stable_sort(indicies+group.first, indicies+group.second, compare);
}

// sorts IDs inside groups by the key column, reorder the column via a row extraction into a temporary 
// DenseMatrix and then scan for groups of duplicates to be tie broken by another subsequent key column
template<typename VTCol>
struct MultiColumnIDSort {
    static void apply(const Frame * arg, DenseMatrix<size_t> *&idx, std::vector<std::pair<size_t, size_t>> &groups, bool ascending, size_t colIdx, DCTX(ctx)) {
        auto col = arg->getColumn<VTCol>(colIdx);
        columnIDSort(idx, col, groups, ascending, ctx);
        DenseMatrix<VTCol> * tmp = nullptr;
        extractRow(tmp, col, idx, ctx);
        columnGroupScan(groups, tmp, ctx);
        DataObjectFactory::destroy(tmp);
    }
};

template<typename VTCol>
struct ColumnIDSort {
    static void apply(const Frame * arg, DenseMatrix<size_t> *&idx, std::vector<std::pair<size_t, size_t>> &groups, bool ascending, size_t colIdx, DCTX(ctx)) {
        auto col = arg->getColumn<VTCol>(colIdx);
        columnIDSort(idx, col, groups, ascending, ctx);
    }
};

template <> struct Order<Frame> {
    static void apply(Frame *& res, const Frame * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
        size_t numRows = arg->getNumRows();
        if (arg == nullptr || colIdxs == nullptr || numColIdxs == 0 || ascending == nullptr) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }
        
        auto idx = DataObjectFactory::create<DenseMatrix<size_t>>(numRows, 1, false);
        auto indicies = idx->getValues();
        std::iota(indicies, indicies+numRows, 0);
        
        std::vector<std::pair<size_t, size_t>> groups;
        groups.push_back(std::make_pair(0, numRows));
            
        if (numColIdxs > 1) {
            for (size_t i = 0; i < numColIdxs-1; i++) {
                DeduceValueTypeAndExecute<MultiColumnIDSort>::apply(arg->getSchema()[colIdxs[i]], arg, idx, groups, ascending[i], colIdxs[i], ctx);
            }
        }

        // efficient last sort pass OR finalizing the groups vector for further use
        size_t colIdx = colIdxs[numColIdxs-1];
        if (groupsRes == nullptr) {
            DeduceValueTypeAndExecute<ColumnIDSort>::apply(arg->getSchema()[colIdx], arg, idx, groups, ascending[numColIdxs-1], colIdx, ctx);
        } else {
            DeduceValueTypeAndExecute<MultiColumnIDSort>::apply(arg->getSchema()[colIdx], arg, idx, groups, ascending[numColIdxs-1], colIdx, ctx);
            if (groups.front().first == 0 && groups.front().second == numRows) {
                groups.clear();
            }
            groupsRes->insert(groupsRes->end(), groups.begin(), groups.end());
        }

        //applying the final object ID permutation (result of the sorting procedure) to the frame via a row extraction
        extractRow(res, arg, idx, ctx);
        DataObjectFactory::destroy(idx);
    }
};