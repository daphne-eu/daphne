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

ValueTypeCode getDaphneType(duckdb::PhysicalType phys){
    std::stringstream error("");
    switch(phys){
        case duckdb::PhysicalType::BOOL:
            error << "duckDbSql(...) does not yet support bool Types.\n";
            throw std::runtime_error(error.str());
            break;
        case duckdb::PhysicalType::INT8:
            return ValueTypeUtils::codeFor<int8_t>;
        case duckdb::PhysicalType::INT16: //todo
            return ValueTypeUtils::codeFor<int32_t>;
        case duckdb::PhysicalType::INT32:
            return ValueTypeUtils::codeFor<int32_t>;
        case duckdb::PhysicalType::INT64:
            return ValueTypeUtils::codeFor<int64_t>;
        case duckdb::PhysicalType::INT128:  //todo
            return ValueTypeUtils::codeFor<int64_t>;
        case duckdb::PhysicalType::UINT8:
            return ValueTypeUtils::codeFor<uint8_t>;
        case duckdb::PhysicalType::UINT16:
            return ValueTypeUtils::codeFor<uint32_t>;
        case duckdb::PhysicalType::UINT32:
            return ValueTypeUtils::codeFor<uint32_t>;
        case duckdb::PhysicalType::UINT64:
            return ValueTypeUtils::codeFor<uint64_t>;
        case duckdb::PhysicalType::FLOAT:
            return ValueTypeUtils::codeFor<float>;
        case duckdb::PhysicalType::DOUBLE:
            return ValueTypeUtils::codeFor<double>;
        case duckdb::PhysicalType::VARCHAR:
            error << "innerJoin(...) does not yet support String Types.\n";
            throw std::runtime_error(error.str());
        default:
            error << "innerJoin(...). The physical return type from the "
                << " DuckDB Query is not supported. The Type is: ";
            error << duckdb::TypeIdToString(phys) << "\n";
            throw std::runtime_error(error.str());
    }
}


void createDataChunk(
    duckdb::DataChunk &dc,
    const Frame* arg
){
    const size_t numCols = arg->getNumCols();
    const size_t numRows = arg->getNumRows();

    std::vector<duckdb::LogicalType> types_ddb;
    for(size_t i = 0; i < numCols; i++){
        ValueTypeCode type = arg->getColumnType(i);
        types_ddb.push_back(getDuckType(type));
    }
    dc.InitializeEmpty(types_ddb);
    for(size_t i = 0; i < numCols; i++){
        duckdb::Vector temp(types_ddb[i], (duckdb::data_ptr_t)arg->getColumnRaw(i));
        dc.data[i].Reference(temp);
    }
    dc.SetCardinality(numRows);
}


template<typename VTCol1, typename VTCol2>
void fillResultFrameColumn(
    Frame *& res,
    uint8_t * data,
    const size_t column,
    const size_t r_f_s,
    const size_t r_f_e,
    const size_t r_dc_s,
    const size_t r_dc_e,
    const size_t offset = 0
){
    const size_t add = 1 + offset;

    VTCol1* res_c = res->getColumn<VTCol1>(column)->getValues();
    VTCol2* data_c = (VTCol2*) data;

    size_t r_f = r_f_s;
    size_t r_dc = r_dc_s;
    while(r_f < r_f_e && r_dc < r_dc_e){
        res_c[r_f] = (VTCol1)data_c[r_dc];
        r_f++;
        r_dc+= add;
    }
}


void fillFrame(Frame*& res, duckdb::DataChunk& data, size_t row = 0, size_t column = 0){

    size_t col_max_dc = data.ColumnCount();
    size_t row_max_dc = data.size();
    size_t col_max_f = res->getNumCols();
    size_t row_max_f = res->getNumRows();

    size_t c_dc = 0;
    size_t c_f = column;
    while(c_dc < col_max_dc && c_f < col_max_f){
        // std::cout << "\t" << c_dc << "," << c_f << "\t" << col_max_dc << "," << col_max_f << std::endl;
        duckdb::Vector dc_vec = move(data.data[c_dc]);
        duckdb::data_ptr_t dc_raw = dc_vec.GetData();

        duckdb::PhysicalType phys = dc_vec.GetType().InternalType();

        switch(phys){
            case duckdb::PhysicalType::INT8:
                fillResultFrameColumn<int8_t, int8_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT16: //todo
                fillResultFrameColumn<int32_t, int16_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT32:
                fillResultFrameColumn<int32_t, int32_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT64:
                fillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT128:
                fillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 1, row_max_dc*2, 1
                );  //todo
                break;
            case duckdb::PhysicalType::UINT8:
                fillResultFrameColumn<uint8_t, uint8_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT16:
                fillResultFrameColumn<uint32_t, uint16_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT32:
                fillResultFrameColumn<uint32_t, int32_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT64:
                fillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::FLOAT:
                fillResultFrameColumn<float, float>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::DOUBLE:
                fillResultFrameColumn<double, double>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
        }

        c_dc++;
        c_f++;
    }
}


#ifdef USE_DUCKSC

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
    std::cout << "innerJoin, DuckDB with SC access!" << std::endl;
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


// //start loading data into DuckDB
//     std::vector<duckdb::LogicalType> types_l;
//     std::vector<duckdb::LogicalType> types_r;
    duckdb::DataChunk dc_l;
    duckdb::DataChunk dc_r;

    createDataChunk(dc_l, lhs);
    createDataChunk(dc_r, rhs);

//Data transfer finished
    //For the InnerJoin we need 4 to 5 DataChunks. two with the data. two for the Join Condition. one for the results

//Datat prep for join
    duckdb::DataChunk dc_join_l;
    duckdb::DataChunk dc_join_r;

    std::vector<duckdb::LogicalType> types_l_j;
    std::vector<duckdb::LogicalType> types_r_j;
    types_l_j.push_back(getDuckType(lhs->getColumnType(on_l)));
    types_r_j.push_back(getDuckType(rhs->getColumnType(on_r)));
    dc_join_l.InitializeEmpty(types_l_j);
    dc_join_r.InitializeEmpty(types_r_j);
    dc_join_l.data[0].Reference(dc_l.data[on_l]);
    dc_join_r.data[0].Reference(dc_r.data[on_r]);
    dc_join_l.SetCardinality(dc_l);
    dc_join_r.SetCardinality(dc_r);

    duckdb::SelectionVector sel_l(dc_l.size());
    duckdb::SelectionVector sel_r(dc_r.size());

    duckdb::idx_t l_t = 0, r_t = 0;

//join
    std::vector<duckdb::JoinCondition> condition ;
    duckdb::JoinCondition equal;//This innerJoin is always an Equi-Join
    equal.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
    condition.push_back(move(equal));
    //Exectution
    size_t match_count = duckdb::NestedLoopJoinInner::Perform(l_t, r_t, dc_join_l, dc_join_r, sel_l, sel_r, condition);

//Result
    dc_l.Slice(sel_l, match_count);
    dc_r.Slice(sel_r, match_count);
    //WE NEED TO FLATTEN! Otherwise all the data is still there and we can copy the wrong result. (DICTIONARY)
    dc_l.Flatten();
    dc_r.Flatten();
    //here we could use another data_chunk for fusing the two results together.
    //but i guess we could forgo this and just create a dataframe that we fill.
    res = DataObjectFactory::create<Frame>(match_count, totalCols, schema, newlabels, false);

    //Tiled reading. Saves a creation of one DataChunk or modification of one.
    fillFrame(res, dc_l);
    fillFrame(res, dc_r, 0, dc_l.ColumnCount());
}



#else if USE_DUCKAPI


void fillDuckDbTable(
    duckdb::Connection &con,
    const Frame *arg,
    const char * name
) {
    duckdb::DataChunk dc_append;
    createDataChunk(dc_append, arg);
    con.Append(con.TableInfo(name), dc_append);
}


void createDuckDbTable(
    duckdb::Connection &con,
    const Frame *arg,
    const char * name
) {
    const size_t numCols = arg->getNumCols();
    std::stringstream s_stream;
    s_stream << "CREATE TABLE " << name << "(";

    for(size_t i = 0; i < numCols; i++){
        ValueTypeCode type = arg->getColumnType(i);
        std::string label = arg->getLabels()[i];
        duckdb::LogicalType ddb_type = getDuckType(type);
        s_stream << label << " " << ddb_type.ToString();
        if(i < numCols - 1){
            s_stream << ", ";
        }
    }
    s_stream << ")";
    std::cout << s_stream.str() << std::endl;
    con.Query(s_stream.str());
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
    std::cout << "Not yet implemented!" std::endl;

    std::string t_lhs_name = "lhs", t_rhs_name = "rhs";
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    createDuckDbTable(con, lhs, t_lhs_name);
    fillDuckDbTable(con, lhs, t_lhs_name);
    duckdb::shared_ptr<duckdb::Relation> t_lhs = con.Table(t_lhs_name);

    createDuckDbTable(con, rhs, t_rhs_name);
    fillDuckDbTable(con, rhs, t_rhs_name);
    duckdb::shared_ptr<duckdb::Relation> t_rhs = con.Table(t_rhs_name);

    std::string condition = lhsOn + " = " + rhsOn;

    duckdb::shared_ptr<duckdb::Relation> join = t_lhs->Join(t_rhs, condition);
    duckdb::unique_ptr<duckdb::QueryResult> result = join->Execute();

    //MERGING AND LOADING IN. IT WOULD BE POSSIBLE TO TILE IT USING the While LOOP
    duckdb::unique_ptr<duckdb::DataChunk> mainChunk = result->Fetch();
    duckdb::unique_ptr<duckdb::DataChunk> nextChunk = result->Fetch();

    while(nextChunk != nullptr){
        mainChunk->Append((const_cast<duckdb::DataChunk&>(*nextChunk)), true);
        nextChunk = result->Fetch();
    }

    fillFrame(res, mainChunk);
}

#endif //DUCKDB Type


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
    std::cout << "innerJoin, DAPHNE!" << std::endl;

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
