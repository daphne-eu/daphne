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


template<typename VT>
void innerJoinSetColumn(
    DenseMatrix<VT> * res,
    const DenseMatrix<VT> * arg,
    const size_t *hit_list,
    const size_t result_size,
    DCTX(ctx)
){
    for(size_t i = 0; i < result_size; i++){
        const VT argValue = arg->get(hit_list[i], 0);
        res->set(i, 0, argValue);
    }
}

template<typename VT>
void innerJoinSetGetColumn(
    Frame *&res,
    const Frame * arg,
    const size_t *hit_list,
    const size_t result_size,
    const int64_t toCol,
    const int64_t fromCol,
    DCTX(ctx)
){
    // std::cout << "\t\tVor Fehler\n";
    ValueTypeCode vtcArg, vtcRes;
    vtcArg = arg->getColumnType(fromCol);
    vtcRes = res->getColumnType(toCol);
    if(vtcArg != ValueTypeUtils::codeFor<VT> || vtcRes != ValueTypeUtils::codeFor<VT>){
        throw std::runtime_error("Inner Join Type mismatch!");
    }
    innerJoinSetColumn<VT>(res->getColumn<VT>(toCol), arg->getColumn<VT>(fromCol), hit_list, result_size, ctx);
}

void innerJoinSetColumnWise(
    Frame *&res,
    const Frame * arg,
    const size_t *hit_list,
    const size_t result_size,
    const size_t offset,
    DCTX(ctx)
) {
    // std::cout << "\tVor Fehler\n";
    const size_t numColArg = arg->getNumCols();

    for(size_t i = 0; i < numColArg; i++){
        ValueTypeCode vtcArg;
        vtcArg = arg->getColumnType(i);

        switch(vtcArg){
            case ValueTypeCode::SI8:
                innerJoinSetGetColumn<int8_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::SI32:
                innerJoinSetGetColumn<int32_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::SI64:
                innerJoinSetGetColumn<int64_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::UI8:
                innerJoinSetGetColumn<uint8_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::UI32:
                innerJoinSetGetColumn<uint32_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::UI64:
                innerJoinSetGetColumn<uint64_t>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::F32:
                innerJoinSetGetColumn<float>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
            case ValueTypeCode::F64:
                innerJoinSetGetColumn<double>(res, arg, hit_list, result_size, i + offset, i, ctx);
                break;
        }
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



template<typename VT>
size_t innerJoinProbeColumn(
    // arguments
    VT* argLhs_raw,
    VT* argRhs_raw,
    size_t *& hit_list_l, size_t *& hit_list_r,
    const size_t lhs_row, const size_t rhs_row,
    // context
    DCTX(ctx)
){
    size_t hit = 0;
    VT* argLhs = argLhs_raw;
    VT* argRhs = argRhs_raw;

    for(size_t l = 0; l < lhs_row; l++){
        for(size_t r = 0; r < rhs_row; r++){
            const VT l_val = argLhs[l];
            const VT r_val = argRhs[r];
            hit_list_l[hit] = l;
            hit_list_r[hit] = r;
            hit += (int) (l_val == r_val);
            // std::cout << l << " == " << r << " ?" << hit <<"\n";
        }
    }
    return hit;
}

template<typename VT>
size_t innerJoinProbeGetColumn(
    const Frame * lhs, const Frame * rhs,
    size_t *& hit_list_l, size_t *& hit_list_r,
    const char * lhsOn, const char * rhsOn,
    DCTX(ctx)
){
    return innerJoinProbeColumn<VT>(
        (VT*)lhs->getColumnRaw(lhs->getColumnIdx(lhsOn)),
        (VT*)rhs->getColumnRaw(rhs->getColumnIdx(rhsOn)),
        hit_list_l,
        hit_list_r,
        lhs->getNumRows(),
        rhs->getNumRows(),
        ctx
    );
}


size_t innerJoinProbe(
    // input frames
    const Frame * lhs, const Frame * rhs,
    // input column names
    size_t *& hit_list_l, size_t *& hit_list_r,
    //label
    const char * lhsOn, const char * rhsOn,
    // context
    DCTX(ctx)
){
    ValueTypeCode vtcLHS = lhs->getColumnType(lhsOn);
    ValueTypeCode vtcRHS = rhs->getColumnType(rhsOn);
    if(vtcLHS != vtcRHS){
        throw std::runtime_error("Inner Join Probe Type mismatch!");
    }

    switch(vtcLHS){
        case ValueTypeCode::SI8:
            return innerJoinProbeGetColumn<int8_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::SI32:
            return innerJoinProbeGetColumn<int32_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::SI64:
            return innerJoinProbeGetColumn<int64_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::UI8:
            return innerJoinProbeGetColumn<uint8_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::UI32:
            return innerJoinProbeGetColumn<uint32_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::UI64:
            return innerJoinProbeGetColumn<uint64_t>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::F32:
            return innerJoinProbeGetColumn<float>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        case ValueTypeCode::F64:
            return innerJoinProbeGetColumn<double>(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);
        default:
            throw std::runtime_error("Inner Join Probe Type unknown!");
    }

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
    //std::cout << "innerJoin, DAPHNE!" << std::endl;

    // Find out the value types of the columns to process.
    ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);
    ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);

    // Perhaps check if res already allocated.
    const size_t numRowRhs = rhs->getNumRows();
    const size_t numRowLhs = lhs->getNumRows();
    const size_t totalRows = numRowRhs > numRowLhs? numRowRhs: numRowLhs;
    const size_t numColRhs = rhs->getNumCols();
    const size_t numColLhs = lhs->getNumCols();
    const size_t totalCols = numColRhs + numColLhs;
    const std::string * oldlabels_l = lhs->getLabels();
    const std::string * oldlabels_r = rhs->getLabels();

    int64_t col_idx_res = 0;
    int64_t row_idx_res = 0;
    int64_t row_result_size = 0;

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
    };



    size_t *hit_list_l = new size_t[totalRows];
    size_t *hit_list_r = new size_t[totalRows];

    //std::cout << "start Probe\n";
    row_result_size = innerJoinProbe(lhs, rhs, hit_list_l, hit_list_r, lhsOn, rhsOn, ctx);

    //std::cout << "creat result\n";
    // Creating Result Frame
    res = DataObjectFactory::create<Frame>(row_result_size, totalCols, schema, newlabels, false);

    //std::cout << "start copy\n";

    innerJoinSetColumnWise(
        res,
        lhs,
        hit_list_l,
        row_result_size,
        0,
        ctx
    );

    innerJoinSetColumnWise(
        res,
        rhs,
        hit_list_r,
        row_result_size,
        numColLhs,
        ctx
    );

    delete[] hit_list_r;
    delete[] hit_list_l;

//    std::cout << "size: " << row_idx_res << std::endl;
    //res->shrinkNumRows(row_idx_res);
}
#endif //SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
