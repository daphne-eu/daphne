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
#include <type_traits>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Order {
    static void apply(DTRes *& res, const DTArg * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

// Note that we normally don't pass any arguments after the DaphneContext. In
// this case it is only okay because groupsRes has a default and is meant to be
// used only by other kernels, not from DaphneDSL.
template<class DTRes, class DTArg>
void order(DTRes *& res, const DTArg * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
    Order<DTRes,DTArg>::apply(res, arg, colIdxs, numColIdxs, ascending, numAscending, returnIdx, ctx, groupsRes);
}

// ****************************************************************************
// Functions called by multiple template specializations
// ****************************************************************************

// sorts input idx DenseMatrix within the index ranges in the input groups vector on the values of the input column (column with id colIdx in DenseMatrix arg; id 0 for column matrices)
template<typename VTIdx, typename VT>
void columnIDSort(DenseMatrix<VTIdx> *&idx, const DenseMatrix<VT>* arg, size_t colIdx, std::vector<std::pair<VTIdx, VTIdx>> &groups, bool ascending, DCTX(ctx)) {
    VTIdx * indices = idx->getValues();
    const VT * values = arg->getValues();
    const size_t rowSkip = arg->getRowSkip();
    auto compare = ascending ?
        std::function{[&values, rowSkip, colIdx](VTIdx i, VTIdx j) {
            return values[i * rowSkip + colIdx] < values[j * rowSkip + colIdx];
        }} :
        std::function{[&values, rowSkip, colIdx](VTIdx i, VTIdx j) {
            return values[i * rowSkip + colIdx] > values[j * rowSkip + colIdx];}};
    for (const auto &group : groups)
        std::stable_sort(indices+group.first, indices+group.second, compare);
}

// function overload for generic Matrix type
template<typename VTIdx, typename VT>
void columnIDSort(DenseMatrix<VTIdx> *&idx, const Matrix<VT>* arg, size_t colIdx, std::vector<std::pair<VTIdx, VTIdx>> &groups, bool ascending, DCTX(ctx)) {
    VTIdx * indices = idx->getValues();
    auto compare = ascending ?
        std::function{[&arg, colIdx](VTIdx i, VTIdx j) {
            return arg->get(i, colIdx) < arg->get(j, colIdx);
        }} :
        std::function{[&arg, colIdx](VTIdx i, VTIdx j) {
            return arg->get(i, colIdx) > arg->get(j, colIdx);}};
    for (const auto &group : groups)
        std::stable_sort(indices+group.first, indices+group.second, compare);
}

template<typename VT>
VT* nextWithRowskip(VT*& first, VT*& last, const size_t rowSkip) {
    for (VT* next = first; next != last; next+=rowSkip) {
        if (*next != *first)
            return next;
    }
    return last;
}

// function overload for generic Matrix type
template<typename VT>
size_t nextWithRowskip(const size_t firstIdx, const size_t lastIdx, const size_t colIdx, Matrix<VT> * arg) {
    const VT firstVal = arg->get(firstIdx, colIdx);
    for (size_t nextIdx = firstIdx; nextIdx < lastIdx; ++nextIdx) {
        if (arg->get(nextIdx, colIdx) != firstVal)
            return nextIdx;
    }
    return lastIdx;
}

// scans a column for groups of duplicates (stored as VTIdx pairs forming index ranges)
// performed only within the index ranges in the input groups vector on the values of the input column (DenseMatrix)
// replaces the groups in the input vector with the groups found during the scan
template<typename VTIdx, typename VT>
void columnGroupScan(std::vector<std::pair<VTIdx, VTIdx>> &groups, DenseMatrix<VT>* col, const size_t colIdx, DCTX(ctx)) {
    const size_t numOldGroups = groups.size();
    VT * values = col->getValues();
    const size_t rowSkip = col->getRowSkip();
    for (size_t g = 0; g < numOldGroups; g++) {
        VT * first = values + groups[g].first * rowSkip + colIdx;
        VT * last = values + groups[g].second * rowSkip + colIdx;
        while (first != last) {
            VT * next = nextWithRowskip<VT>(first, last, rowSkip);
            if ((size_t) (next - first) > rowSkip)
                groups.push_back(std::make_pair((first-values-colIdx)/rowSkip, (next-values-colIdx)/rowSkip));
            first = next;
        }
    }
    groups.erase(groups.begin(),groups.begin()+numOldGroups);
}

// function overload for generic Matrix type
template<typename VTIdx, typename VT>
void columnGroupScan(std::vector<std::pair<VTIdx, VTIdx>> &groups, Matrix<VT>* col, const size_t colIdx, DCTX(ctx)) {
    const size_t numOldGroups = groups.size();
    for (size_t g = 0; g < numOldGroups; ++g) {
        size_t currentRowIdx = groups[g].first;
        size_t lastRowIdx = groups[g].second;
        while (currentRowIdx != lastRowIdx) {
            size_t nextIdx = nextWithRowskip<VT>(currentRowIdx, lastRowIdx, colIdx, col);
            if ((nextIdx - currentRowIdx) > 1)
                groups.push_back(std::make_pair(currentRowIdx, nextIdx));
            currentRowIdx = nextIdx;
        }
    }
    groups.erase(groups.begin(), groups.begin()+numOldGroups);
}

// sorts IDs inside groups by the key column, reorder the column via a row extraction into a temporary
// DenseMatrix and then scan for groups of duplicates to be tie broken by another subsequent key column
template<typename VTIdx, typename DTCol>
void multiColumnIDSort(DenseMatrix<VTIdx> *&idx, const DTCol* col, size_t colIdx, std::vector<std::pair<VTIdx, VTIdx>> &groups, bool ascending, DCTX(ctx)){
    columnIDSort(idx, col, colIdx, groups, ascending, ctx);
    DTCol * tmp = nullptr;
    extractRow<DTCol, DTCol, VTIdx>(tmp, col, idx, ctx);
    columnGroupScan(groups, tmp, colIdx, ctx);
    DataObjectFactory::destroy(tmp);
}

// ----------------------------------------------------------------------------
//  Frame order structs
// ----------------------------------------------------------------------------

template<typename VTCol>
struct ColumnIDSort {
    static void apply(const Frame * arg, DenseMatrix<size_t> *&idx, std::vector<std::pair<size_t, size_t>> &groups, bool ascending, size_t colIdx, DCTX(ctx)) {
        columnIDSort(idx, arg->getColumn<VTCol>(colIdx), 0, groups, ascending, ctx);
    }
};

// sorts IDs inside groups by the key column, reorder the column via a row extraction into a temporary 
// DenseMatrix and then scan for groups of duplicates to be tie broken by another subsequent key column
template<typename VTCol>
struct MultiColumnIDSort {
    static void apply(const Frame * arg, DenseMatrix<size_t> *&idx, std::vector<std::pair<size_t, size_t>> &groups, bool ascending, size_t colIdx, DCTX(ctx)) {
        multiColumnIDSort(idx, arg->getColumn<VTCol>(colIdx), 0, groups, ascending, ctx);
    }
};

struct OrderFrame {
    static void apply(DenseMatrix<size_t> *& idx, const Frame * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, std::vector<std::pair<size_t, size_t>> * groupsRes, DCTX(ctx)) {
        size_t numRows = arg->getNumRows();
        idx = DataObjectFactory::create<DenseMatrix<size_t>>(numRows, 1, false);
        auto indices = idx->getValues();
        std::iota(indices, indices+numRows, 0);
        
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
            groupsRes->insert(groupsRes->end(), groups.begin(), groups.end());
        }
    }
};

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

template <> struct Order<Frame, Frame> {
    static void apply(Frame *& res, const Frame * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
        if (arg == nullptr || colIdxs == nullptr || numColIdxs == 0 || ascending == nullptr || returnIdx) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }
        DenseMatrix<size_t>* idx = nullptr;
        OrderFrame::apply(idx, arg, colIdxs, numColIdxs, ascending, numAscending, groupsRes, ctx);
        extractRow(res, arg, idx, ctx);
        DataObjectFactory::destroy(idx);
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- Frame
// ----------------------------------------------------------------------------

template <typename VTRes> struct Order<DenseMatrix<VTRes>, Frame> {
    static void apply(DenseMatrix<VTRes> *& res, const Frame * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
        if (arg == nullptr || colIdxs == nullptr || numColIdxs == 0 || ascending == nullptr || !returnIdx) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }
        DenseMatrix<size_t>* idx = nullptr;
        OrderFrame::apply(idx, arg, colIdxs, numColIdxs, ascending, numAscending, groupsRes, ctx);
        res = idx;
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct Order<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
        size_t numRows = arg->getNumRows();
        if (arg == nullptr || colIdxs == nullptr || numColIdxs == 0 || ascending == nullptr ||
            (returnIdx == false && !std::is_same<VTRes, VTArg>::value) ||
            (returnIdx == true && !std::is_same<VTRes, size_t>::value)
        ) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }

        auto idx = DataObjectFactory::create<DenseMatrix<size_t>>(numRows, 1, false);
        auto indices = idx->getValues();
        std::iota(indices, indices+numRows, 0);
        std::vector<std::pair<size_t, size_t>> groups;
        groups.push_back(std::make_pair(0, numRows));

        if (numColIdxs > 1) {
            for (size_t i = 0; i < numColIdxs-1; i++) {
                multiColumnIDSort(idx, arg, colIdxs[i], groups, ascending[i], ctx);
            }
        }

        if (groupsRes == nullptr) {
            columnIDSort(idx, arg, colIdxs[numColIdxs-1], groups, ascending[numColIdxs-1], ctx);
        } else {
            multiColumnIDSort(idx, arg, colIdxs[numColIdxs-1], groups, ascending[numColIdxs-1], ctx);
            groupsRes->insert(groupsRes->end(), groups.begin(), groups.end());
        }

        if (returnIdx)
        {
           res = (DenseMatrix<VTRes>*) idx;
        } else {
            if constexpr(std::is_same<VTArg, VTRes>::value)
                extractRow(res, arg, idx, ctx);
            else
                // When returnIdx is false, then VTRes and VTArg should be the same, so this should not happen.
                throw std::runtime_error("when returnIdx is false, then VTRes and VTArg should be the same");
            DataObjectFactory::destroy(idx);
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct Order<Matrix<VTRes>, Matrix<VTArg>> {
    static void apply(Matrix<VTRes> *& res, const Matrix<VTArg> * arg, size_t * colIdxs, size_t numColIdxs, bool * ascending, size_t numAscending, bool returnIdx, DCTX(ctx), std::vector<std::pair<size_t, size_t>> * groupsRes = nullptr) {
        size_t numRows = arg->getNumRows();

        if (arg == nullptr || colIdxs == nullptr || numColIdxs == 0 || ascending == nullptr ||
            (returnIdx == false && !std::is_same<VTRes, VTArg>::value)
        ) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }

        auto idx = DataObjectFactory::create<DenseMatrix<size_t>>(numRows, 1, false);
        auto indices = idx->getValues();
        std::iota(indices, indices+numRows, 0);
        std::vector<std::pair<size_t, size_t>> groups;
        groups.push_back(std::make_pair(0, numRows));

        if (numColIdxs > 1) {
            for (size_t i = 0; i < numColIdxs-1; ++i)
                multiColumnIDSort(idx, arg, colIdxs[i], groups, ascending[i], ctx);
        }

        if (groupsRes == nullptr) {
            columnIDSort(idx, arg, colIdxs[numColIdxs-1], groups, ascending[numColIdxs-1], ctx);
        } else {
            multiColumnIDSort(idx, arg, colIdxs[numColIdxs-1], groups, ascending[numColIdxs-1], ctx);
            groupsRes->insert(groupsRes->end(), groups.begin(), groups.end());
        }

        if (returnIdx) {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);
            else if (res->getNumRows() != numRows || res->getNumCols() != 1)
                throw std::runtime_error("Order: given res has wrong shape");

            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r)
                res->append(r, 0, static_cast<VTRes>(indices[r]));
            res->finishAppend();
        } else {
            if constexpr(std::is_same<VTArg, VTRes>::value)
                extractRow(res, arg, idx, ctx);
            else
                // When returnIdx is false, then VTRes and VTArg should be the same, so this should not happen.
                throw std::runtime_error("when returnIdx is false, then VTRes and VTArg should be the same");
        }

        DataObjectFactory::destroy(idx);
    }
};
