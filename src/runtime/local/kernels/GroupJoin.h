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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include <cstddef>
#include <cstdint>

// TODO This entire implementation of the group-join is very inefficient and
// there are numerous opportunities for improvement. However, currently, we
// just need it to work.

// ****************************************************************************
// Utility function
// ****************************************************************************
// TODO Maybe this should be a kernel on its own.

template <typename VTLhs, typename VTRhs, typename VTAgg, typename VTTid>
void groupJoinCol(
    // results
    Frame *&res, DenseMatrix<VTTid> *&resLhsTid,
    // arguments
    const DenseMatrix<VTLhs> *argLhs, const DenseMatrix<VTRhs> *argRhs, const DenseMatrix<VTAgg> *argAgg,
    // context
    DCTX(ctx)) {
    if (argLhs->getNumCols() != 1)
        throw std::runtime_error("parameter argLhs must be a single-column matrix");
    if (argRhs->getNumCols() != 1)
        throw std::runtime_error("parameter argRhs must be a single-column matrix");
    if (argAgg->getNumCols() != 1)
        throw std::runtime_error("parameter argAgg must be a single-column matrix");
    if (argRhs->getNumRows() != argAgg->getNumRows())
        throw std::runtime_error("parameters argRhs and argAgg must have the same number of rows");

    std::unordered_map<VTLhs, std::tuple<size_t, VTAgg, bool>> ht;

    // ------------------------------------------------------------------------
    // Build phase on argLhs.
    // ------------------------------------------------------------------------
    const size_t numArgLhs = argLhs->getNumRows();
    for (size_t i = 0; i < numArgLhs; i++)
        ht.emplace(argLhs->get(i, 0), std::make_tuple(i, 0, false));

    // ------------------------------------------------------------------------
    // Probe phase on argRhs.
    // ------------------------------------------------------------------------
    const size_t numArgRhs = argRhs->getNumRows();
    for (size_t i = 0; i < numArgRhs; i++) {
        auto it = ht.find(argRhs->get(i, 0));
        if (it != ht.end()) {
            std::get<1>(it->second) += argAgg->get(i, 0);
            std::get<2>(it->second) = true;
        }
    }

    // ------------------------------------------------------------------------
    // Output phase.
    // ------------------------------------------------------------------------

    // Determine the number of output rows.
    size_t numRes = 0;
    for (auto it = ht.begin(); it != ht.end(); it++)
        if (std::get<2>(it->second))
            numRes++;

    // Create the output data objects.
    if (res == nullptr) {
        ValueTypeCode schema[] = {ValueTypeUtils::codeFor<VTLhs>, ValueTypeUtils::codeFor<VTAgg>};
        res = DataObjectFactory::create<Frame>(numRes, 2, schema, nullptr, false);
    }
    auto resLhs = res->getColumn<VTLhs>(0);
    auto resAgg = res->getColumn<VTAgg>(1);
    if (resLhsTid == nullptr)
        resLhsTid = DataObjectFactory::create<DenseMatrix<VTTid>>(numRes, 1, false);

    // Write the results.
    size_t pos = 0;
    for (auto it = ht.begin(); it != ht.end(); it++)
        if (std::get<2>(it->second)) {
            resLhs->set(pos, 0, it->first);
            resAgg->set(pos, 0, std::get<1>(it->second));
            resLhsTid->set(pos, 0, std::get<0>(it->second));
            pos++;
        }

    // Free intermediate data objects.
    // TODO This is not possible at the moment due to a bug in Frame (ownership
    // of the underlying data is not shared correctly).
    //    DataObjectFactory::destroy(resLhs);
    //    DataObjectFactory::destroy(resAgg);
}

template <typename VTLhs, typename VTRhs, typename VTAgg, typename VTTid>
void groupJoinColIf(
    // value type known only at run-time
    ValueTypeCode vtcLhs, ValueTypeCode vtcRhs, ValueTypeCode vtcAgg,
    // results
    Frame *&res, DenseMatrix<VTTid> *&resLhsTid,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn, const char *rhsOn, const char *rhsAgg,
    // context
    DCTX(ctx)) {
    if (vtcLhs == ValueTypeUtils::codeFor<VTLhs> && vtcRhs == ValueTypeUtils::codeFor<VTRhs> &&
        vtcAgg == ValueTypeUtils::codeFor<VTAgg>) {
        groupJoinCol<VTLhs, VTRhs, VTAgg, VTTid>(res, resLhsTid, lhs->getColumn<VTLhs>(lhsOn),
                                                 rhs->getColumn<VTRhs>(rhsOn), rhs->getColumn<VTAgg>(rhsAgg), ctx);
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTLhsTid>
void groupJoin(
    // results
    Frame *&res, DenseMatrix<VTLhsTid> *&lhsTid,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn, const char *rhsOn, const char *rhsAgg,
    // context
    DCTX(ctx)) {
    // Find out the value types of the columns to process.
    ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);
    ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);
    ValueTypeCode vtcRhsAgg = rhs->getColumnType(rhsAgg);

    // Call the groupJoin-kernel on columns for the actual combination of
    // value types.
    // Repeat this for all type combinations...
    groupJoinColIf<int64_t, int64_t, double, VTLhsTid>(vtcLhsOn, vtcRhsOn, vtcRhsAgg, res, lhsTid, lhs, rhs, lhsOn,
                                                       rhsOn, rhsAgg, ctx);
    groupJoinColIf<int64_t, int64_t, int64_t, VTLhsTid>(vtcLhsOn, vtcRhsOn, vtcRhsAgg, res, lhsTid, lhs, rhs, lhsOn,
                                                        rhsOn, rhsAgg, ctx);

    // Set the column labels of the result frame.
    std::string labels[] = {lhsOn, std::string("SUM(") + rhsAgg + std::string(")")};
    res->setLabels(labels);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H