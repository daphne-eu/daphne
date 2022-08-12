#ifndef SRC_RUNTIME_LOCAL_KERNELS_DUCKDBSQL_H
#define SRC_RUNTIME_LOCAL_KERNELS_DUCKDBSQL_H

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
// Convenience function
// ****************************************************************************

void duckDbSql(
    // results
    Frame *& res,
    // query
    const char * query,
    //Frames (Tables)
    Frame ** tables,
    size_t numTables,
    //Frame Names (Table Names)
    const char ** tableNames,
    size_t numTableNames,
    // context
    DCTX(ctx)
) {
    // // Find out the value types of the columns to process.
    // ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);
    // ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);
    //
    // // Perhaps check if res already allocated.
    // const size_t numRowRhs = rhs->getNumRows();
    // const size_t numRowLhs = lhs->getNumRows();
    // const size_t totalRows = numRowRhs * numRowLhs;
    // const size_t numColRhs = rhs->getNumCols();
    // const size_t numColLhs = lhs->getNumCols();
    // const size_t totalCols = numColRhs + numColLhs;
    // const std::string * oldlabels_l = lhs->getLabels();
    // const std::string * oldlabels_r = rhs->getLabels();
    //
    // int64_t col_idx_res = 0;
    // int64_t row_idx_res = 0;
    //
    // ValueTypeCode schema[totalCols];
    // std::string newlabels[totalCols];
    //
    // // Setting Schema and Labels
    // for(size_t col_idx_l = 0; col_idx_l < numColLhs; col_idx_l++){
    //     schema[col_idx_res] = lhs->getColumnType(col_idx_l);
    //     newlabels[col_idx_res] = oldlabels_l[col_idx_l];
    //     col_idx_res++;
    // }
    // for(size_t col_idx_r = 0; col_idx_r < numColRhs; col_idx_r++){
    //     schema[col_idx_res] = rhs->getColumnType(col_idx_r);
    //     newlabels[col_idx_res] = oldlabels_r[col_idx_r];
    //     col_idx_res++;
    // }
    //
    // // Creating Result Frame
    // res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, newlabels, false);
    //
    // for(size_t row_idx_l = 0; row_idx_l < numRowLhs; row_idx_l++){
    //     for(size_t row_idx_r = 0; row_idx_r < numRowRhs; row_idx_r++){
    //         col_idx_res = 0;
    //         //PROBE ROWS
    //         bool hit = false;
    //         hit = hit || innerJoinProbeIf<int64_t, int64_t>(
    //             vtcLhsOn, vtcRhsOn,
    //             res,
    //             lhs, rhs,
    //             lhsOn, rhsOn,
    //             row_idx_l, row_idx_r,
    //             ctx);
    //         hit = hit || innerJoinProbeIf<double, double>(
    //             vtcLhsOn, vtcRhsOn,
    //             res,
    //             lhs, rhs,
    //             lhsOn, rhsOn,
    //             row_idx_l, row_idx_r,
    //             ctx);
    //         if(hit){
    //             for(size_t idx_c = 0; idx_c < numColLhs; idx_c++){
    //                 innerJoinSet<int64_t>(
    //                     schema[col_idx_res],
    //                     res,
    //                     lhs,
    //                     row_idx_res,
    //                     col_idx_res,
    //                     row_idx_l,
    //                     idx_c,
    //                     ctx
    //                 );
    //                 innerJoinSet<double>(
    //                     schema[col_idx_res],
    //                     res,
    //                     lhs,
    //                     row_idx_res,
    //                     col_idx_res,
    //                     row_idx_l,
    //                     idx_c,
    //                     ctx
    //                 );
    //                 col_idx_res++;
    //             }
    //             for(size_t idx_c = 0; idx_c < numColRhs; idx_c++){
    //                 innerJoinSet<int64_t>(
    //                     schema[col_idx_res],
    //                     res,
    //                     rhs,
    //                     row_idx_res,
    //                     col_idx_res,
    //                     row_idx_r,
    //                     idx_c,
    //                     ctx
    //                 );
    //
    //                 innerJoinSet<double>(
    //                     schema[col_idx_res],
    //                     res,
    //                     rhs,
    //                     row_idx_res,
    //                     col_idx_res,
    //                     row_idx_r,
    //                     idx_c,
    //                     ctx
    //                 );
    //                 col_idx_res++;
    //             }
    //             row_idx_res++;
    //         }
    //     }
    // }
    // res->shrinkNumRows(row_idx_res);




    // Perhaps check if res already allocated.
    const size_t totalRows = 0; //numRowRhs * numRowLhs;
    const size_t totalCols = 1;//numColRhs + numColLhs;


    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];


    schema[0] = ValueTypeCode::SI64;
    newlabels[0] = "res";

    // Creating Result Frame
    res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, newlabels, false);
}
#endif //SRC_RUNTIME_LOCAL_KERNELS_DUCKDBSQL_H
