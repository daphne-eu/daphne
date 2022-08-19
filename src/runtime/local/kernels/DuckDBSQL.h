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

#include <sstream>

// ****************************************************************************
// Convenience function
// ****************************************************************************
#ifdef USE_DUCKDB
#include <duckdb.hpp>




template<typename VTCol>
VTCol getValue(
    const DenseMatrix<VTCol> *arg,
    const size_t row
){
    const VTCol value = arg->get(row, 0);
    return value;
}

template <typename VTCol>
VTCol getValueFromFrame(
    const Frame * arg,
    const size_t row,
    const size_t col
){
    return getValue<VTCol>(arg->getColumn<VTCol>(col), row);
}


void fillDuckDbTable(
    duckdb::Connection &con,
    const Frame *arg,
    const char * name
) {
    const size_t numCols = arg->getNumCols();
    const size_t numRows = arg->getNumRows();
    duckdb::Appender appender(con, name);
    for(size_t row = 0; row < numRows; row++){

        appender.BeginRow();
        for(size_t col = 0; col < numCols; col++){
            const ValueTypeCode type = arg->getColumnType(col);
            switch (type) {
                case ValueTypeCode::SI8:
                    appender.Append(getValueFromFrame<int8_t>(arg, row, col));
                    break;
                case ValueTypeCode::SI32:
                    appender.Append(getValueFromFrame<int32_t>(arg, row, col));
                    break;
                case ValueTypeCode::SI64:
                    appender.Append(getValueFromFrame<int64_t>(arg, row, col));
                    break;
                case ValueTypeCode::UI8:
                    appender.Append(getValueFromFrame<uint8_t>(arg, row, col));
                    break;
                case ValueTypeCode::UI32:
                    appender.Append(getValueFromFrame<uint32_t>(arg, row, col));
                    break;
                case ValueTypeCode::UI64:
                    appender.Append(getValueFromFrame<uint64_t>(arg, row, col));
                    break;
                case ValueTypeCode::F32:
                    appender.Append(getValueFromFrame<float>(arg, row, col));
                    break;
                case ValueTypeCode::F64:
                    appender.Append(getValueFromFrame<double>(arg, row, col));
                    break;
                default:
                    std::stringstream error;
                    error << "duckDbSql(...) doesn't support the given"
                        << "ValueType belonging to cpp type name: "
                        << ValueTypeUtils::cppNameForCode(type);
                    throw std::runtime_error(error.str());
            }
        }
        appender.EndRow();

    }
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
        switch (type) {
            case ValueTypeCode::SI8:
                s_stream << label << " TINYINT";
                break;
            case ValueTypeCode::SI32:
                s_stream << label << " INTEGER";
                break;
            case ValueTypeCode::SI64:
                s_stream << label << " BIGINT";
                break;
            case ValueTypeCode::UI8:
                s_stream << label << " UTINYINT";
                break;
            case ValueTypeCode::UI32:
                s_stream << label << " UINTEGER";
                break;
            case ValueTypeCode::UI64:
                s_stream << label << " UBIGINT";
                break;
            case ValueTypeCode::F32:
                s_stream << label << " REAL";
                break;
            case ValueTypeCode::F64:
                s_stream << label << " DOUBLE";
                break;
            default:
                std::stringstream error;
                error << "duckDbSql(...) doesn't support the given"
                    << "ValueType belonging to cpp type name: "
                    << ValueTypeUtils::cppNameForCode(type);
                throw std::runtime_error(error.str());
        }
        if(i < numCols - 1){
            s_stream << ", ";
        }
    }
    s_stream << ")";
    con.Query(s_stream.str());
    std::cout << s_stream.str() << std::endl;
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

    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    std::cout<<std::endl << query << std::endl;
    for(size_t i = 0; i < numTableNames && i < numTables; i++){
        createDuckDbTable(con, tables[i], tableNames[i]);
        fillDuckDbTable(con, tables[i], tableNames[i]);
    }
    duckdb::unique_ptr<duckdb::MaterializedQueryResult> result = con.Query(query);

    if(!result->success){
        std::stringstream error;
        error << "duckDbSql(...): DuckDB query execution unsuccessful: " << query;
        error << "\nDuckDB reports: " << result->error;
        throw std::runtime_error(error.str());
    }

    std::cout << result->ToString() << std::endl;

    //TODO: MaterializedQueryResult -> Frame


    std::vector<duckdb::LogicalType> ret_types = result->types;
    std::vector<std::string> ret_names = result->names;


    const size_t totalCols = ret_types.size() == ret_names.size()? ret_types.size(): -1;


    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    std::stringstream error("");
    bool error_raised = false;
    for(size_t i = 0; i < totalCols; i++){
        newlabels[i] = ret_names[i];
        duckdb::LogicalType logi = ret_types[i];
        duckdb::PhysicalType phys = logi.InternalType();
        switch(phys){
            case duckdb::PhysicalType::BOOL:
                error << "duckDbSql(...) does not yet support bool Types.\n";
                error_raised = true;
                break;
            case duckdb::PhysicalType::INT8:
                schema[i] = ValueTypeUtils::codeFor<int8_t>;
                break;
            case duckdb::PhysicalType::INT16: //todo
                schema[i] = ValueTypeUtils::codeFor<int32_t>;
                break;
            case duckdb::PhysicalType::INT32:
                schema[i] = ValueTypeUtils::codeFor<int32_t>;
                break;
            case duckdb::PhysicalType::INT64:
                schema[i] = ValueTypeUtils::codeFor<int64_t>;
                break;
            case duckdb::PhysicalType::INT128:  //todo
                schema[i] = ValueTypeUtils::codeFor<int64_t>;
                break;
            case duckdb::PhysicalType::UINT8:
                schema[i] = ValueTypeUtils::codeFor<uint8_t>;
                break;
            case duckdb::PhysicalType::UINT16:
                schema[i] = ValueTypeUtils::codeFor<uint32_t>;
                break;
            case duckdb::PhysicalType::UINT32:
                schema[i] = ValueTypeUtils::codeFor<uint32_t>;
                break;
            case duckdb::PhysicalType::UINT64:
                schema[i] = ValueTypeUtils::codeFor<uint64_t>;
                break;
            case duckdb::PhysicalType::HALF_FLOAT:  //todo
                schema[i] = ValueTypeUtils::codeFor<float>;
                break;
            case duckdb::PhysicalType::FLOAT:
                schema[i] = ValueTypeUtils::codeFor<float>;
                break;
            case duckdb::PhysicalType::DOUBLE:
                schema[i] = ValueTypeUtils::codeFor<double>;
                break;
            case duckdb::PhysicalType::STRING:  //todo
                error << "duckDbSql(...) does not yet support String Types.\n";
                error_raised = true;
                break;
            default:
                error << "duckDbSql(...). The physical return type from the "
                    << " DuckDB Query is not supported. The LogicalType is: ";
                error << duckdb::LogicalTypeIdToString(logi.id()) << "\n";
                error_raised = true;
        }
    }
    if(error_raised){
        throw std::runtime_error(error.str());
    }

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




    // schema[0] = ValueTypeCode::SI64;
    // newlabels[0] = "res";
    // Creating Result Frame
}
#else //NO DUCKDB
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
    throw std::runtime_error("Use of duckDbSql(...) without DuckDB (--duckdb).");
}
#endif

#endif //SRC_RUNTIME_LOCAL_KERNELS_DUCKDBSQL_H
