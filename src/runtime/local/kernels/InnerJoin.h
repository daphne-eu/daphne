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
#include <unordered_set>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Helper functions
// ****************************************************************************

template<typename VTCol>
void innerJoinSetValue(
    DenseMatrix<VTCol> * res,
    const DenseMatrix<VTCol> * arg,
    const int64_t targetRow,
    const int64_t fromRow,
    DCTX(ctx)
){
    const VTCol argValue = arg->get(fromRow, 0);
    res->set(targetRow, 0, argValue);
}

template<typename VTCol>
void innerJoinSet(
    ValueTypeCode vtcType,
    Frame *&res,
    const Frame * arg,
    const int64_t toRow,
    const int64_t toCol,
    const int64_t fromRow,
    const int64_t fromCol,
    DCTX(ctx)
) {
    if(vtcType == ValueTypeUtils::codeFor<VTCol>){
        innerJoinSetValue<VTCol>(
            res->getColumn<VTCol>(toCol),
            arg->getColumn<VTCol>(fromCol),
            toRow,
            fromRow,
            ctx
        );
    }
}

template<typename VTLhs, typename VTRhs>
bool innerJoinEqual(
    // results
    Frame *& res,
    // arguments
    const DenseMatrix<VTLhs> * argLhs,
    const DenseMatrix<VTRhs> * argRhs,
    const int64_t targetLhs,
    const int64_t targetRhs,
    // context
    DCTX(ctx)
){
    const VTLhs l = argLhs->get(targetLhs, 0);
    const VTRhs r = argRhs->get(targetRhs, 0);
    return l == r;
}

template<typename VTLhs, typename VTRhs>
bool innerJoinProbeIf(
    // value type known only at run-time
    ValueTypeCode vtcLhs,
    ValueTypeCode vtcRhs,
    // results
    Frame *& res,
    // input frames
    const Frame * lhs, const Frame * rhs,
    // input column names
    const char * lhsOn, const char * rhsOn,
    // input rows
    const int64_t targetL, const int64_t targetR,
    // context
    DCTX(ctx)
){
    if(vtcLhs == ValueTypeUtils::codeFor<VTLhs> && vtcRhs == ValueTypeUtils::codeFor<VTRhs>) {
        return innerJoinEqual<VTLhs, VTRhs>(
                res,
                lhs->getColumn<VTLhs>(lhsOn),
                rhs->getColumn<VTRhs>(rhsOn),
                targetL,
                targetR,
                ctx
        );
    }
    return false;
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

void innerJoin(
    // results
    Frame *& res,
    // input frames
    const Frame * lhs, const Frame * rhs,
    // input column names
    const char * lhsOn, const char * rhsOn,
    // context
    DCTX(ctx)
) {
    // Find out the value types of the columns to process.
    ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);
    ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);

    // Perhaps check if res already allocated.
    const size_t numRowRhs = rhs->getNumRows();
    const size_t numRowLhs = lhs->getNumRows();
    const size_t totalRows = numRowRhs * numRowLhs;
    const size_t numColRhs = rhs->getNumCols();
    const size_t numColLhs = lhs->getNumCols();
    const size_t totalCols = numColRhs + numColLhs;
    const std::string * oldlabels_l = lhs->getLabels();
    const std::string * oldlabels_r = rhs->getLabels();

    int64_t col_idx_res = 0;
    int64_t row_idx_res = 0;

    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    // Setting Schema and Labels
    for(size_t col_idx_l = 0; col_idx_l < numColLhs; col_idx_l++){
        schema[col_idx_res] = lhs->getColumnType(col_idx_l);
        newlabels[col_idx_res] = oldlabels_l[col_idx_l];
        col_idx_res++;
    }
    for(size_t col_idx_r = 0; col_idx_r < numColRhs; col_idx_r++){
        schema[col_idx_res] = rhs->getColumnType(col_idx_r);
        newlabels[col_idx_res] = oldlabels_r[col_idx_r];
        col_idx_res++;
    }

    // Creating Result Frame
    res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, newlabels, false);

    for(size_t row_idx_l = 0; row_idx_l < numRowLhs; row_idx_l++){
        for(size_t row_idx_r = 0; row_idx_r < numRowRhs; row_idx_r++){
            col_idx_res = 0;
            //PROBE ROWS
            bool hit = false;
            hit = hit || innerJoinProbeIf<int64_t, int64_t>(
                vtcLhsOn, vtcRhsOn,
                res,
                lhs, rhs,
                lhsOn, rhsOn,
                row_idx_l, row_idx_r,
                ctx);
            hit = hit || innerJoinProbeIf<double, double>(
                vtcLhsOn, vtcRhsOn,
                res,
                lhs, rhs,
                lhsOn, rhsOn,
                row_idx_l, row_idx_r,
                ctx);
            if(hit){
                for(size_t idx_c = 0; idx_c < numColLhs; idx_c++){
                    innerJoinSet<int64_t>(
                        schema[col_idx_res],
                        res,
                        lhs,
                        row_idx_res,
                        col_idx_res,
                        row_idx_l,
                        idx_c,
                        ctx
                    );
                    innerJoinSet<double>(
                        schema[col_idx_res],
                        res,
                        lhs,
                        row_idx_res,
                        col_idx_res,
                        row_idx_l,
                        idx_c,
                        ctx
                    );
                    col_idx_res++;
                }
                for(size_t idx_c = 0; idx_c < numColRhs; idx_c++){
                    innerJoinSet<int64_t>(
                        schema[col_idx_res],
                        res,
                        rhs,
                        row_idx_res,
                        col_idx_res,
                        row_idx_r,
                        idx_c,
                        ctx
                    );

                    innerJoinSet<double>(
                        schema[col_idx_res],
                        res,
                        rhs,
                        row_idx_res,
                        col_idx_res,
                        row_idx_r,
                        idx_c,
                        ctx
                    );
                    col_idx_res++;
                }
                row_idx_res++;
            }
        }
    }
    res->shrinkNumRows(row_idx_res);
}
#endif //SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
