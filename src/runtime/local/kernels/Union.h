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
#include <runtime/local/kernels/DuckDBHelper.h>


//Checks if the Frames have the same column count and column type.
void checkFrames(
    Frame ** frames,
    size_t numTables
){
    for(size_t i = 1; i < numTables; i++){
        if(frames[0]->getNumCols() != frames[i]->getNumCols()){
            std::stringstream error;
            error << "Union(...): Frames have unequal column count.";
            throw std::runtime_error(error.str());
        }
        for(size_t k = 1; k < frames[0]->getNumCols(); k++){
            ValueTypeCode type1 = frames[0]->getColumnType(k);
            ValueTypeCode type2 = frames[i]->getColumnType(k);
            if(type1 != type2){
                std::stringstream error;
                error << "Union(...): Frames have different column types.";
                throw std::runtime_error(error.str());
            }
        }
    }
}


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
    std::cout << "Union, DuckDB" << std::endl;
//OPEN CONNECTION
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

//CREATING VARIABLES FOR TRANSFER AND UNION
    duckdb::shared_ptr<duckdb::Relation> tables[numTables];
    duckdb::shared_ptr<duckdb::Relation> unionR;

//CHECKING FRAMES
    checkFrames(frames, numTables);

//LOADING DATA INTO DUCKDB
    for(size_t i = 0; i < numTables; i++){
        std::stringstream table_name_stream;
        table_name_stream << "table_" << i;
        std::string table_name = table_name_stream.str();
        std::cout << table_name << std::endl;
        ddb_CreateAndFill(con, frames[i], table_name.c_str());
        tables[i] = con.Table(table_name.c_str());
    }

//BUILD EXECUTION
    unionR = tables[0];
    for(size_t i = 1; i < numTables; i++){
        unionR = unionR->Union(tables[i]);
    }

//EXECUTION
    duckdb::unique_ptr<duckdb::QueryResult> result = unionR->Execute();

//CHECK EXECUTION FOR ERRORS
    if(result->HasError()){
        std::stringstream error;
        error << "Union(...): DuckDB Union execution unsuccessful. "
            << "DuckDB reports: " << result->GetError();
        throw std::runtime_error(error.str());
    }


//PREPARE FRAMECREATION
    std::vector<duckdb::LogicalType> ret_types = result->types;
    std::vector<std::string> ret_names = result->names;

    const size_t totalCols = ret_types.size() == ret_names.size()? ret_types.size(): 0;
    ValueTypeCode schema[totalCols];
    std::string newlabels[totalCols];

    for(size_t i = 0; i < totalCols; i++){
        newlabels[i] = ret_names[i];
        duckdb::LogicalType logi = ret_types[i];
        duckdb::PhysicalType phys = logi.InternalType();
        schema[i] = ddb_GetDaphneType(phys);
    }

//CREATE FRAME AND TRANSFER DATA BACK
    ddb_FillResultFrame(res, result, schema, newlabels, totalCols);

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
