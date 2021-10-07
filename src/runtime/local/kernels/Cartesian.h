// def Daphne_CartesianOp : Daphne_Op<"cartesian"> {
//     let arguments = (ins Frame:$lhs, Frame:$rhs); //let arguments = (ins Variadic<Frame>:$args); // since i don't know how to use variadic it got replaced..
//     let results = (outs Frame:$res);
// }

// def Daphne_SemiJoinOp : Daphne_Op<"semiJoin", [
//     DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
//     DeclareOpInterfaceMethods<InferTypesOpInterface>
// ]> {
//     let arguments = (ins Frame:$lhs, Frame:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn);
//     let results = (outs Frame:$res, MatrixOf<[Size]>:$lhsTids);
// }

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CARTESIAN_H
#define SRC_RUNTIME_LOCAL_KERNELS_CARTESIAN_H

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

template<typename VTCol>
void cartesianSetValue(
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
void cartesianSet(
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
        cartesianSetValue<VTCol>(
            res->getColumn<VTCol>(toCol),
            arg->getColumn<VTCol>(fromCol),
            toRow,
            fromRow,
            ctx
        );
    }
}

void cartesian(
        Frame *& res,
        const Frame * lhs, const Frame * rhs,
        DCTX(ctx)
) {
    const size_t numRowRhs = rhs->getNumRows();
    const size_t numRowLhs = lhs->getNumRows();
    const size_t totalRows = numRowRhs * numRowLhs;
    const size_t numColRhs = rhs->getNumCols();
    const size_t numColLhs = lhs->getNumCols();
    const size_t totalCols = numColRhs + numColLhs;
    const std::string * oldlabels_l = lhs->getLabels();
    const std::string * oldlabels_r = rhs->getLabels();

    int64_t row_idx_res = 0;
    int64_t col_idx_res = 0;
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

            for(size_t idx_c = 0; idx_c < numColLhs; idx_c++){
                cartesianSet<int64_t>(
                    schema[col_idx_res],
                    res,
                    lhs,
                    row_idx_res,
                    col_idx_res,
                    row_idx_l,
                    idx_c,
                    ctx
                );
                cartesianSet<double>(
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
                cartesianSet<int64_t>(
                    schema[col_idx_res],
                    res,
                    rhs,
                    row_idx_res,
                    col_idx_res,
                    row_idx_r,
                    idx_c,
                    ctx
                );
                cartesianSet<double>(
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

#endif //SRC_RUNTIME_LOCAL_KERNELS_CARTESIAN_H
