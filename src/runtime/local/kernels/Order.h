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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ORDER_H
#define SRC_RUNTIME_LOCAL_KERNELS_ORDER_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/ExtractRow.h>

#include <vector>
#include <algorithm>
#include <numeric>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Order {
    static void apply(DTRes *& res, const DTRes * arg, size_t * colIdxs, size_t numKeyCols, bool * ascending, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void order(DTRes *& res, const DTRes * arg, size_t * colIdxs, size_t numKeyCols, bool * ascending, DCTX(ctx)) {
    Order<DTRes>::apply(res, arg, colIdxs, numKeyCols, ascending, ctx);
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
template<typename VTIdx, typename VT>
void multiColumnIDSort(DenseMatrix<VTIdx> *&idx, const DenseMatrix<VT>* col, std::vector<std::pair<VTIdx, VTIdx>> &groups, bool ascending, DCTX(ctx)){
    columnIDSort(idx, col, groups, ascending, ctx);
    DenseMatrix<VT> * tmp = nullptr;
    extractRow<DenseMatrix<VT>, DenseMatrix<VT>, VTIdx>(tmp, col, idx, ctx);
    columnGroupScan(groups, tmp, ctx);
    DataObjectFactory::destroy(tmp);
} 

template <> struct Order<Frame> {
    static void apply(Frame *& res, const Frame * arg, size_t * colIdxs, size_t numKeyCols, bool * ascending, DCTX(ctx)) {
        size_t numRows = arg->getNumRows();
        if (arg == nullptr || colIdxs == nullptr || numKeyCols == 0 || ascending == nullptr) {
            throw std::runtime_error("order-kernel called with invalid arguments");
        }

        size_t colIdx;
        auto idx = DataObjectFactory::create<DenseMatrix<size_t>>(numRows, 1, false);
        auto indicies = idx->getValues();
        std::iota(indicies, indicies+numRows, 0);
        std::vector<std::pair<size_t, size_t>> groups;
        groups.push_back(std::make_pair(0, numRows));
        
        if (numKeyCols > 1) {
            for (size_t i = 0; i < numKeyCols-1; i++) {
                colIdx = colIdxs[i];
                switch(arg->getColumnType(colIdx)) {
                    // TODO Memory leak (getColumn(), see #222).
                    case ValueTypeCode::F64: multiColumnIDSort(idx, arg->getColumn<double>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::F32: multiColumnIDSort(idx, arg->getColumn<float>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::SI64: multiColumnIDSort(idx, arg->getColumn<int64_t>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::SI32: multiColumnIDSort(idx, arg->getColumn<int32_t>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::SI8 : multiColumnIDSort(idx, arg->getColumn<int8_t>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::UI64: multiColumnIDSort(idx, arg->getColumn<uint64_t>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::UI32: multiColumnIDSort(idx, arg->getColumn<uint32_t>(colIdx), groups, ascending[i], ctx); break;
                    case ValueTypeCode::UI8 : multiColumnIDSort(idx, arg->getColumn<uint8_t>(colIdx), groups, ascending[i], ctx); break;
                    default: throw std::runtime_error("unknown value type code");
                }
            }
        }

        colIdx = colIdxs[numKeyCols-1];
        switch(arg->getColumnType(colIdx)) {
            // TODO Memory leak (getColumn(), see #222).
            case ValueTypeCode::F64: columnIDSort(idx, arg->getColumn<double>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            case ValueTypeCode::F32: columnIDSort(idx, arg->getColumn<float>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            case ValueTypeCode::SI64: columnIDSort(idx, arg->getColumn<int64_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            case ValueTypeCode::SI32: columnIDSort(idx, arg->getColumn<int32_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break; 
            case ValueTypeCode::SI8 : columnIDSort(idx, arg->getColumn<int8_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break; 
            case ValueTypeCode::UI64: columnIDSort(idx, arg->getColumn<uint64_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            case ValueTypeCode::UI32: columnIDSort(idx, arg->getColumn<uint32_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            case ValueTypeCode::UI8 : columnIDSort(idx, arg->getColumn<uint8_t>(colIdx), groups, ascending[numKeyCols-1], ctx); break;
            default: throw std::runtime_error("unknown value type code");
        }

        //applying the final object ID permutation (result of the sorting procedure) to the frame via a row extraction
        extractRow<Frame, Frame, size_t>(res, arg, idx, ctx);
        DataObjectFactory::destroy(idx);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_ORDER_H