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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SEMIJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_SEMIJOIN_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include <cstddef>
#include <cstdint>

// TODO This entire implementation of the semi-join is very inefficient and
// there are numerous opportunities for improvement. However, currently, we
// just need it to work.

// ****************************************************************************
// Utility function
// ****************************************************************************
// TODO Maybe this should be a kernel on its own.

template <typename VTLhs, typename VTRhs, typename VTTid>
void semiJoinCol(
    // results
    Frame *&res, DenseMatrix<VTTid> *&resLhsTid,
    // arguments
    const DenseMatrix<VTLhs> *argLhs, const DenseMatrix<VTRhs> *argRhs,
    // result size
    int64_t numRowRes,
    // context
    DCTX(ctx)) {
    if (argLhs->getNumCols() != 1)
        throw std::runtime_error("parameter argLhs must be a single-column matrix");
    if (argRhs->getNumCols() != 1)
        throw std::runtime_error("parameter argRhs must be a single-column matrix");

    std::unordered_set<VTRhs> hs;

    // ------------------------------------------------------------------------
    // Build phase on argRhs.
    // ------------------------------------------------------------------------

    const size_t numArgRhs = argRhs->getNumRows();
    for (size_t i = 0; i < numArgRhs; i++)
        hs.emplace(argRhs->get(i, 0));

    // ------------------------------------------------------------------------
    // Probe phase on argLhs.
    // ------------------------------------------------------------------------

    const size_t numArgLhs = argLhs->getNumRows();

    // Create the output data objects.
    if (res == nullptr) {
        ValueTypeCode schema[] = {ValueTypeUtils::codeFor<VTLhs>};
        const size_t resSize = numRowRes == -1 ? numArgLhs : numRowRes;
        res = DataObjectFactory::create<Frame>(resSize, 1, schema, nullptr, false);
    }
    auto resLhs = res->getColumn<VTLhs>(0);
    if (resLhsTid == nullptr) {
        const size_t resLhsTidSize = numRowRes == -1 ? numArgLhs : numRowRes;
        resLhsTid = DataObjectFactory::create<DenseMatrix<VTTid>>(resLhsTidSize, 1, false);
    }

    size_t pos = 0;
    for (size_t i = 0; i < numArgLhs; i++) {
        const VTLhs vLhs = argLhs->get(i, 0);
        if (hs.count(vLhs)) {
            resLhs->set(pos, 0, vLhs);
            resLhsTid->set(pos, 0, i);
            pos++;
        }
    }

    res->shrinkNumRows(pos);
    resLhsTid->shrinkNumRows(pos);

    // Free intermediate data objects.
    // TODO This is not possible at the moment due to a bug in Frame (ownership
    // of the underlying data is not shared correctly).
    //    DataObjectFactory::destroy(resLhs);
}

template <typename VTLhs, typename VTRhs, typename VTTid>
void semiJoinColIf(
    // value type known only at run-time
    ValueTypeCode vtcLhs, ValueTypeCode vtcRhs,
    // results
    Frame *&res, DenseMatrix<VTTid> *&resLhsTid,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn, const char *rhsOn,
    // result size
    int64_t numRowRes,
    // context
    DCTX(ctx)) {
    if (vtcLhs == ValueTypeUtils::codeFor<VTLhs> && vtcRhs == ValueTypeUtils::codeFor<VTRhs>) {
        semiJoinCol<VTLhs, VTRhs, VTTid>(res, resLhsTid, lhs->getColumn<VTLhs>(lhsOn), rhs->getColumn<VTRhs>(rhsOn),
                                         numRowRes, ctx);
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTLhsTid>
void semiJoin(
    // results
    Frame *&res, DenseMatrix<VTLhsTid> *&lhsTid,
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
    ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);

    // Call the semiJoin-kernel on columns for the actual combination of
    // value types.
    // Repeat this for all type combinations...
    semiJoinColIf<int64_t, int64_t, VTLhsTid>(vtcLhsOn, vtcRhsOn, res, lhsTid, lhs, rhs, lhsOn, rhsOn, numRowRes, ctx);
    semiJoinColIf<int64_t, int64_t, VTLhsTid>(vtcLhsOn, vtcRhsOn, res, lhsTid, lhs, rhs, lhsOn, rhsOn, numRowRes, ctx);

    // Set the column labels of the result frame.
    std::string labels[] = {lhsOn};
    res->setLabels(labels);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_SEMIJOIN_H