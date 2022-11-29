#ifndef SRC_RUNTIME_LOCAL_KERNELS_DUCKDBHELPER_H
#define SRC_RUNTIME_LOCAL_KERNELS_DUCKDBHELPER_H

#ifdef USE_DUCKDB

#include <duckdb.hpp>
#include <duckdb/main/appender.hpp>

#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/selection_vector.hpp>
#include <duckdb/common/constants.hpp>

#include <iterator>
#include <algorithm>

//a function to fix column nameing for duckdb. (A . in the table namen leads to an error) so we replace it with _
void ddb_Convert(std::string& x){
    auto it = std::find(x.begin(), x.end(), '.');
    if(it < x.end()){
        size_t pos = it - x.begin();
        x.replace(pos, 1, "_");
    }
}


//Returns the DuckDB LogicalType for a given DAPHNE Type Code
duckdb::LogicalType ddb_GetDuckType(ValueTypeCode type){
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

//Returns the DAPHNE Type Code for a given PhysicalType of DuckDB
ValueTypeCode ddb_GetDaphneType(duckdb::PhysicalType phys){
    std::stringstream error("");
    switch(phys){
        case duckdb::PhysicalType::BOOL:
            error << "DAPHNE does not yet support bool Types for "
                <<"ddb_GetDaphneType(...).\n";
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
            error << "DAPHNE does not yet support string Types for "
                << "ddb_GetDaphneType(...).\n";
            throw std::runtime_error(error.str());
        default:
            error << "ddb_GetDaphneType(...). The physical return type from "
                << "DuckDB is not supported. The Type is: ";
            error << duckdb::TypeIdToString(phys) << "\n";
            throw std::runtime_error(error.str());
    }
}

//Creates a DuckDB DataChunk from the DAPHNE Frame
void ddb_CreateDataChunk(
    duckdb::DataChunk &dc,
    const Frame* arg
){
    const size_t numCols = arg->getNumCols();
    const size_t numRows = arg->getNumRows();

    std::vector<duckdb::LogicalType> types_ddb;
    for(size_t i = 0; i < numCols; i++){
        ValueTypeCode type = arg->getColumnType(i);
        types_ddb.push_back(ddb_GetDuckType(type));
    }
    dc.InitializeEmpty(types_ddb);
    for(size_t i = 0; i < numCols; i++){
        duckdb::Vector temp(types_ddb[i], (duckdb::data_ptr_t)arg->getColumnRaw(i));
        dc.data[i].Reference(temp);
    }
    dc.SetCardinality(numRows);
}

//Copies Data from a charArray into a Frame. At a given Column
// VTCOL1 Type of Frame VTCOL2 Type of DataChunk
template<typename VTCol1, typename VTCol2>
void ddb_FillResultFrameColumn(
    Frame *& res,
    uint8_t * data,	//DATA TO BE COPIED (DataChunk data)
    const size_t column,	//WHERE TO PLACE IT IN THE FRAME
    const size_t r_f_s,		//starting row for Frame
    const size_t r_f_e,		//end row +1 for Frame
    const size_t r_dc_s,	//starting row for DataChunk data
    const size_t r_dc_e,	//end row +1 for DataChunk data
    const size_t offset = 0	//Offset if the Datachunk contained int128_t
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

//copies the data from the DataChunk into the Frame. starting at the Frame <row> and <column>
void ddb_FillFrame(
    Frame*& res,
    duckdb::DataChunk& data,
    size_t row = 0,
    size_t column = 0
) {

    size_t col_max_dc = data.ColumnCount();
    size_t row_max_dc = data.size();
    size_t col_max_f = res->getNumCols();
    size_t row_max_f = res->getNumRows();

    size_t c_dc = 0;
    size_t c_f = column;
    while(c_dc < col_max_dc && c_f < col_max_f){
        duckdb::Vector dc_vec = move(data.data[c_dc]);
        duckdb::data_ptr_t dc_raw = dc_vec.GetData();

        duckdb::PhysicalType phys = dc_vec.GetType().InternalType();

        switch(phys){
            case duckdb::PhysicalType::INT8:
                ddb_FillResultFrameColumn<int8_t, int8_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT16: //todo
                ddb_FillResultFrameColumn<int32_t, int16_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT32:
                ddb_FillResultFrameColumn<int32_t, int32_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT64:
                ddb_FillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::INT128:
                ddb_FillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 1, row_max_dc*2, 1
                );  //todo
                break;
            case duckdb::PhysicalType::UINT8:
                ddb_FillResultFrameColumn<uint8_t, uint8_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT16:
                ddb_FillResultFrameColumn<uint32_t, uint16_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT32:
                ddb_FillResultFrameColumn<uint32_t, int32_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::UINT64:
                ddb_FillResultFrameColumn<int64_t, int64_t>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::FLOAT:
                ddb_FillResultFrameColumn<float, float>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
            case duckdb::PhysicalType::DOUBLE:
                ddb_FillResultFrameColumn<double, double>(
                    res, dc_raw, c_f, row, row_max_f, 0, row_max_dc
                );
                break;
        }

        c_dc++;
        c_f++;
    }
}

//We Fill the Frame with every DataChunk seperately
void ddb_VectorizedFill(
    Frame *& res,
    duckdb::unique_ptr<duckdb::QueryResult> &query_result, //doesn't give us the res lenght;
    ValueTypeCode * schema,
    std::string * labels,
    size_t totalCols
){
    size_t totalRows = 0;

    std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunks;
    duckdb::unique_ptr<duckdb::DataChunk> nextChunk;

    while((nextChunk = query_result->Fetch()) != nullptr){
        chunks.push_back(move(nextChunk));
        totalRows += nextChunk->size();
    }

	res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, labels, false);

    size_t pos = 0;
    for(duckdb::unique_ptr<duckdb::DataChunk>& chunk : chunks){
        ddb_FillFrame(res, *chunk, pos);
        pos += chunk->size();
    }
}

//We Append every DataChunk onto the next and than fill.
void ddb_MergeFill(
    Frame *& res,
    duckdb::unique_ptr<duckdb::QueryResult> &query_result, //doesn't give us the res lenght
    ValueTypeCode * schema,
    std::string * labels,
    size_t totalCols
) {
    size_t totalRows = 0;
    duckdb::unique_ptr<duckdb::DataChunk> mainChunk = query_result->Fetch();
    duckdb::unique_ptr<duckdb::DataChunk> nextChunk;

    while((nextChunk = query_result->Fetch()) != nullptr){
        mainChunk->Append((const_cast<duckdb::DataChunk&>(*nextChunk)), true);
    }
    totalRows = mainChunk->size();

    res = DataObjectFactory::create<Frame>(totalRows, totalCols, schema, labels, false);

    ddb_FillFrame(res, *mainChunk);
}

void ddb_FillResultFrame(
    Frame *& res,
    duckdb::unique_ptr<duckdb::QueryResult> &query_result, //doesn't give us the res lenght
    ValueTypeCode * schema,
    std::string * labels,
    size_t totalCols
){
    //Retrieve Result (2 Ways. VECTORISED and STANDARD(?) (Names need Work))
    #ifdef USE_DUCKVECTORISED
        //QueryResult doesn't has a Possibility to get the row count.
        //In this case: We have to Fetch all the DataChunks and add up there lenght.
        //Then we store one DataChunk at a time into a Frame.
        ddb_VectorizedFill(res, query_result, schema, labels, totalCols);
    #else //DEFAULTMODE
        //In this case we Fetch all the DataChunks and fit append them together into
        //one DataChunk. This datachunk has a row count and we can load this data into
        //the Frame!
        ddb_MergeFill(res, query_result, schema, labels, totalCols);
    #endif // Result type
}


//Gets the Table from the connection and appends the Frame to the Table.
void ddb_FillDuckDbTable(
    duckdb::Connection &con,
    const Frame *arg,
    const char *name
) {
    duckdb::DataChunk dc_append;
    ddb_CreateDataChunk(dc_append, arg); //Creates a DataChunk to Append
    duckdb::Appender appender(con, name);
    appender.AppendDataChunk(dc_append);
}

//Creates a DuckDB Table so that we can use it for API Access.
//WE HAVE TO ISSUE A CREATE TABLE QUERY.
void ddb_CreateDuckDbTable(
    duckdb::Connection &con,
    const Frame *arg,
    const char *name
) {
    const size_t numCols = arg->getNumCols();
    std::stringstream s_stream;
    s_stream << "CREATE TABLE " << name << "(";

    for(size_t i = 0; i < numCols; i++){
        ValueTypeCode type = arg->getColumnType(i);
        std::stringstream l_stream;
        std::string label = arg->getLabels()[i];
        ddb_Convert(label);
        duckdb::LogicalType ddb_type = ddb_GetDuckType(type);
        s_stream << label << " " << ddb_type.ToString();
        if(i < numCols - 1){
            s_stream << ", ";
        }
    }
    s_stream << ")";
    //std::cout << s_stream.str() << std::endl;
    con.Query(s_stream.str());
}

void ddb_CreateAndFill(
	duckdb::Connection &con,
	const Frame *arg,
	const char *name
){
	ddb_CreateDuckDbTable(con, arg, name);
	ddb_FillDuckDbTable(con, arg, name);
}

#endif //USE DUCKDB
#endif //SRC_RUNTIME_LOCAL_KERNELS_DUCKDBHELPER_H
