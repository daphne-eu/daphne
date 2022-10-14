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

#ifdef USE_DUCKDB

#include <duckdb.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/selection_vector.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/planner/joinside.hpp>
#include <duckdb/execution/nested_loop_join.hpp>

duckdb::LogicalType getDuckType(ValueTypeCode type){
    switch(type){
        case ValueTypeCode::SI8:
            return duckdb::LogicalType(duckdb::LogicalTypeId::TINYINT);
        case ValueTypeCode::SI32:
            return duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
        case ValueTypeCode::SI64:
            return duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
        case ValueTypeCode::UI8:
            return duckdb::LogicalType(duckdb::LogicalTypeId::UTINYINT);
        case ValueTypeCode::UI32:
            return duckdb::LogicalType(duckdb::LogicalTypeId::UINTEGER);
        case ValueTypeCode::UI64:
            return duckdb::LogicalType(duckdb::LogicalTypeId::UBIGINT);
        case ValueTypeCode::F32:
            return duckdb::LogicalType(duckdb::LogicalTypeId::FLOAT);
        case ValueTypeCode::F64:
            return duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
        default:
            std::stringstream error;
            error << "innerJoin.h with DuckDB support doesn't "
                << "support the given ValueType belonging to cpp type name: "
                << ValueTypeUtils::cppNameForCode(type)
                << ". Error in Function getDuckType()";
            throw std::runtime_error(error.str());
    }
}


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
    //general Data
    const size_t numCols_l = lhs->getNumCols();
    const size_t numCols_r = rhs->getNumCols();
    const size_t totalCols = numCols_r + numCols_l;
    const size_t numRow_r = rhs->getNumRows();
    const size_t numRow_l = lhs->getNumRows();
    const size_t on_l = lhs->getColumnIdx(lhsOn);
    const size_t on_r = rhs->getColumnIdx(rhsOn);
    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    int64_t col_idx_res = 0;
    for(size_t col_idx_l = 0; col_idx_l < numCols_l; col_idx_l++){
        schema[col_idx_res] = lhs->getColumnType(col_idx_l);
        newlabels[col_idx_res] = lhs->getLabels()[col_idx_l];
        col_idx_res++;
    }
    for(size_t col_idx_r = 0; col_idx_r < numCols_r; col_idx_r++){
        schema[col_idx_res] = rhs->getColumnType(col_idx_r);
        newlabels[col_idx_res] = rhs->getLabels()[col_idx_r];
        col_idx_res++;
    }

//start loading data into DuckDB
    std::vector<duckdb::LogicalType> types_l;
    std::vector<duckdb::LogicalType> types_r;
    duckdb::DataChunk dc_l;
    duckdb::DataChunk dc_r;

    for(size_t i = 0; i < numCols_l; i++){
        ValueTypeCode type = lhs->getColumnType(i);
        types_l.push_back(getDuckType(type));
    }
    for(size_t i = 0; i < numCols_r; i++){
        ValueTypeCode type = rhs->getColumnType(i);
        types_r.push_back(getDuckType(type));
    }
    dc_l.InitializeEmpty(types_l);
    dc_r.InitializeEmpty(types_r);

    for(size_t i = 0; i < numCols_l; i++){
        duckdb::Vector temp(types_l[i], (duckdb::data_ptr_t)lhs->getColumnRaw(i));
        dc_l.data[i].Reference(temp);
    }
    for(size_t i = 0; i < numCols_r; i++){
        duckdb::Vector temp(types_r[i], (duckdb::data_ptr_t)rhs->getColumnRaw(i));
        dc_r.data[i].Reference(temp);
    }
    dc_l.SetCardinality(numRow_l);
    dc_r.SetCardinality(numRow_r);
//Data transfer finished
    //For the InnerJoin we need 5(?) DataChunks. two with the data. two for the Join Condition. one for the results

//Datat prep for join
    duckdb::DataChunk dc_join_l;
    duckdb::DataChunk dc_join_r;
    std::vector<duckdb::idx_t> join_ons_l;
    join_ons_l.push_back(on_l);
    std::vector<duckdb::idx_t> join_ons_r;
    join_ons_r.push_back(on_r);

    dc_join_l.ReferenceColumns(dc_l, join_ons_l);
    dc_join_r.ReferenceColumns(dc_r, join_ons_r);
    duckdb::SelectionVector sel_l;
    duckdb::SelectionVector sel_r;

    duckdb::idx_t l_t = 0, r_t = 0;
//join
    std::vector<duckdb::JoinCondition> condition;
    duckdb::JoinCondition equal;//This innerJoin is always an Equi-Join
    equal.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
    condition.push_back(equal);
    size_t match_count = duckdb::NestedLoopJoinInner::Perform(l_t, r_t, dc_join_l, dc_join_r, sel_l, sel_r, condition);
//Result
    dc_l.Slice(sel_l, match_count);
    dc_r.Slice(sel_r, match_count);
    //here we could use another data_chunk for fusing the two results together.
    //but i guess we could forgo this and just create a dataframe that we fill.
    res = DataObjectFactory::create<Frame>(match_count, totalCols, schema, newlabels, false);

    col_idx_res = 0;
    for(size_t i = 0; i < numCols_l; i++){
        res->column[col_idx_res] = (int8_t*) dc_l.data[i];
        col_idx_res++;
    }
    for(size_t i = 0; i < numCols_r; i++){
        res->column[col_idx_res] = (int8_t*) dc_r.data[i];
        col_idx_res++;
    }

}



#else //NO DuckDB

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
#endif //DuckDB
#endif //SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
