#ifndef SRC_RUNTIME_LOCAL_KERNELS_UNION_H
#define SRC_RUNTIME_LOCAL_KERNELS_UNION_H

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

#include <sstream>

// ****************************************************************************
// Convenience function
// ****************************************************************************
#ifdef USE_DUCKDB
#include <duckdb.hpp>
#include "duckdb/main/query_result.hpp"
#include <duckdb/main/appender.hpp>


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
            error << "Union.h with DuckDB support doesn't "
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
            error << "duckDbSql(...) does not yet support String Types.\n";
            throw std::runtime_error(error.str());
        default:
            error << "duckDbSql(...). The physical return type from the "
                << " DuckDB Query is not supported. The LogicalType is: ";
            error << duckdb::LogicalTypeIdToString(logi.id()) << "\n";
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

//TODO! we have a better way of doing it now!
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

template<typename VTCol>
void SetValues(
    DenseMatrix<VTCol> * res,
    const duckdb::Vector arg,
    const size_t length
){
    for(size_t row = 0; row < length; row++){
        const VTCol argValue = arg.GetValue(row).GetValue<VTCol>();
        res->set(row, 0, argValue);
    }
    free(arg); //MAYBE?!
}

template<typename VTCol>
void fillResultFrameColumn(
    Frame *& res,
    duckdb::Vector data,
    const size_t col,
    const size_t length
){
    SetValues(
        res->getColumn<VTCol>(col),
        move(data),
        length
    );
}

void fillResultFrame(
    Frame *& res,
    duckdb::unique_ptr<duckdb::DataChunk> chunk,
    ValueTypeCode * schema,
    const size_t width,
    const size_t length
){
    for(size_t col = 0; col < width; col++){
        duckdb::Vector data = move(chunk->data[col]);
        ValueTypeCode vt = schema[col];
        switch (vt) {
            case ValueTypeCode::SI8:
                fillResultFrameColumn<int8_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::SI32:
                fillResultFrameColumn<int32_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::SI64:
                fillResultFrameColumn<int64_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::UI8:
                fillResultFrameColumn<uint8_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::UI32:
                fillResultFrameColumn<uint32_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::UI64:
                fillResultFrameColumn<uint64_t>(res, move(data), col, length);
                break;
            case ValueTypeCode::F32:
                fillResultFrameColumn<float>(res, move(data), col, length);
                break;
            case ValueTypeCode::F64:
                fillResultFrameColumn<double>(res, move(data), col, length);
                break;
            default:
                std::stringstream error;
                error << "duckDbSql(...) in fillResultFrame(...) doesn't "
                    << "support the given ValueType belonging to cpp type name: "
                    << ValueTypeUtils::cppNameForCode(vt);
                throw std::runtime_error(error.str());
        }
    }

}

void checkFrames(
    Frame ** frames,
    size_t numTables
){
    for(size_t i = 1; i < numTables; i++){
        if(frame[0]->getNumCols() != frame[i]->getNumCols()){
            std::stringstream error;
            error << "Union(...): Frames have unequal column count.";
            throw std::runtime_error(error.str());
        }
        for(size_t k = 1; k < frame[0]->getNumCols()){
            ValueTypeCode type1 = frames[0]->getColumnType(x);
            ValueTypeCode type2 = frames[i]->getColumnType(x);
            if(type1 != type2){
                std::stringstream error;
                error << "Union(...): Frames have different column types.";
                throw std::runtime_error(error.str());
            }
        }
    }
}


// void createAndFillDuckDBTable(duckdb::Connection con, Frame frame, const char* name){
//     createDuckDbTable(con, frame, name);
//     fillDuckDbTable(con, frame, name);
// }

//We use the innerjoin method to input datachunk via the connection. than get the table via relations and use union. IMPORTANT! tables need to have the same types and lengths. otherwise empty.
void Union(
    // results
    Frame *& res,
    //Frames (Tables)
    Frame ** frames,
    size_t numTables,
    // context
    DCTX(ctx)
) {
//OPEN CONNECTION AND
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    duckdb::shared_ptr<duckdb::Relation> tables[numTables];
    duckdb::shared_ptr<duckdb::Relation> unionR;
    std::cout<<std::endl << query << std::endl;

//CHECKING FRAMES
    checkFrames(frames, numTables);

//LOAD DATA INTO DUCKDB
    for(size_t i = 0; i < numTables; i++){
        std::string table_name = "table_" + (int8_t)('a' + i);
        createDuckDbTable(con, frames[i], table_name);
        fillDuckDbTable(con, frames[i], table_name);
        tables[i] = con.Table(table_name);
    }

//BUILD EXECUTION
    unionR = tables[0];
    for(size_t i = 1; i < numTables; i++){
        unionR = unionR->Union(tables[i]);
    }
    duckdb::unique_ptr<duckdb::QueryResult> result = unionR->Execute();

//CHECK EXECUTION FOR ERRORS
    if(result->HasError()){
        std::stringstream error;
        error << "duckDbSql(...): DuckDB Union execution unsuccessful: " << query;
        error << "\nDuckDB reports: " << result->GetError();
        throw std::runtime_error(error.str());
    }

//GET DATA AND TRANSFER IT BACK INTO DAPHNE
    std::vector<duckdb::LogicalType> ret_types = result->types;
    std::vector<std::string> ret_names = result->names;

    const size_t totalCols = ret_types.size() == ret_names.size()? ret_types.size(): 0;

    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    for(size_t i = 0; i < totalCols; i++){
        newlabels[i] = ret_names[i];
        duckdb::PhysicalType phys = ret_types[i].InternalType();
        schema[i] = getDaphneType(phys);
    }

//GET DATA FROM DUCKDB TO DAPHNE
    duckdb::unique_ptr<duckdb::DataChunk> mainChunk = result->Fetch();
    duckdb::unique_ptr<duckdb::DataChunk> nextChunk = result->Fetch();

    while(nextChunk != nullptr){
        mainChunk->Append((const_cast<duckdb::DataChunk&>(*nextChunk)), true);
        nextChunk = result->Fetch();
    }

    const size_t totalRows = mainChunk->size();
    res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, newlabels, false);

    fillResultFrame(
        res,
        move(mainChunk),
        schema,
        totalCols,
        totalRows
    );
}


#else //NO DUCKDB
void Union(
    // results
    Frame *& res,
    //Frames (Tables)
    Frame ** tables,
    size_t numTables,
    // context
    DCTX(ctx)
) {
    throw std::runtime_error("Use of Union(...) without DuckDB (--duckdb). At the moment only a DuckDB implementation exists.");
}
#endif

#endif //SRC_RUNTIME_LOCAL_KERNELS_DUCKDBSQL_H
