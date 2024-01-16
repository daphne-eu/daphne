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

/**
 * Add the values from the frame to the final result.
 * Iterate trough all the columns of the frame and add each row
 * that is present in the hit vector.
 * 
 * @param from the source frame
 * @param hit the hit vector for current frame
 * @param col_idx_res the current colum in the result frame
 * 
 * @result result the out frame
*/
template<typename VTCol>
void innerJoinSet(
    Frame * result,
    const Frame * from,
    const std::vector<int> *hit,
    int64_t * col_idx_res

){
    auto numCols = from->getNumCols();

    for (size_t idx_c = 0; idx_c < numCols; idx_c++) {

        auto fromCol = from->getColumn<VTCol>(idx_c);
        auto resCol = result->getColumn<VTCol>(*col_idx_res);
        int resRowIdx = 0;
            for (auto rowIdx : *hit) {
                auto value = fromCol->get(rowIdx, 0);
                resCol->set(resRowIdx, 0, value);
                resRowIdx++;
            }

        (*col_idx_res)++;
    }
}

/**
 * Determine the intersection of the two frames. Create one vector for
 * each frame containing the row index that should be added in the final
 * result.
 * 
 * @param lhs left frame
 * @param rhs right frame
 * @param lhsOn label of the left frame
 * @param rhsOn label of the right frame
 * 
 * @result lHit vector for the left frame
 * @result rHit vector for the right frame
*/
template<typename VTCol>
void innerJoinIntersection(
    std::vector<int> * lHit,
    std::vector<int> * rHit,
    const Frame * lhs,
    const Frame * rhs,
    const char * lhsOn, 
    const char * rhsOn
) {

    auto const lCol = lhs->getColumn<VTCol>(lhsOn);
    auto const rCol = rhs->getColumn<VTCol>(rhsOn);

    auto const numRowLhs = lhs->getNumRows();
    auto const numRowRhs = rhs->getNumRows();

    for (size_t row_idx_l = 0; row_idx_l < numRowLhs; row_idx_l++) {
        for (size_t row_idx_r = 0; row_idx_r < numRowRhs; row_idx_r++) {
            if (lCol->get(row_idx_l, 0) == rCol->get(row_idx_r, 0)) {
                lHit->push_back(row_idx_l);
                rHit->push_back(row_idx_r);
            }
        }
    }
}

/**
 * Helper function to combine the innerJoinIntersection and 
 * innerJoinSet to perform the full inner join.
 * 
 * @param lhs the left frame
 * @param rhs the right frame
 * @param lhsOn label of the left frame
 * @param rhsOn label of the right frame
 * 
 * @result res the final frame
 * @result row_idx_res number of rows of the final frame
*/
template<typename VTCol>
void computeInnerJoin(
    Frame * res,
    const Frame * lhs,
    const Frame * rhs,
    const char * lhsOn, 
    const char * rhsOn,
    int64_t *row_idx_res
) {

    std::vector<int> lHit;
    std::vector<int> rHit;

    int64_t col_idx_res = 0;

    innerJoinIntersection<VTCol>(&lHit, &rHit, lhs, rhs, lhsOn, rhsOn);
    innerJoinSet<VTCol>(res, lhs, &lHit, &col_idx_res);
    innerJoinSet<VTCol>(res, rhs, &rHit, &col_idx_res);

    *(row_idx_res) = lHit.size();

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
    const size_t totalRows = (numRowRhs < numRowLhs) ? numRowLhs : numRowRhs;
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


    if (vtcLhsOn == vtcRhsOn) {
        col_idx_res = 0;
        switch (vtcLhsOn) {
                case ValueTypeCode::SI64: {
                    computeInnerJoin<int64_t>(res, lhs, rhs, lhsOn, rhsOn, &row_idx_res);
                    break;
                }
                case ValueTypeCode::F64: {
                    computeInnerJoin<double>(res, lhs, rhs, lhsOn, rhsOn, &row_idx_res);
                    break;
                }
                default: {
                    break;
                }
        }
    }

    res->shrinkNumRows(row_idx_res);
}
#endif //SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
