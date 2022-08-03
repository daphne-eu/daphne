/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/
//./build.sh --target duckDB_test
#include <tags.h>
#include <vector>
#include <cstdint>
#include <iostream>

#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <catch.hpp>

#include <duckdb.hpp>
#include <sstream>


uint32_t noise(int32_t position, int32_t mod = 10, int32_t seed = 0){
    uint32_t mangled = (uint32_t) position;
    mangled *= 2147483647;
    mangled += seed;
    mangled ^= (mangled >> 13);
    mangled += 1423;
    mangled ^= (mangled << 13);
    mangled *= mangled;
    mangled ^= (mangled >> 13);
    return mangled % mod;
}

// void createTable(Connection con, std::vector<std::vector<int32_t>> value, std::string table, std::vector<std::string> schema, std::vector<std::string> type){
//     std::stringstream createTable;
//     createTable << "CREATE TABLE " << table << "("
//     uint64_t minSize = values.at(0).size();
//     for(uint64_t i = 1; i < values.size() &&  i < schema.size() && i < type.size(); i++){
//         createTable << schema.at(i) << " " << type.at(i) << ", ";
//         if(minSize > values.at(i).size()){
//             minSize = values.at(i).size();
//         }
//     }
//     createTable << schema.at(i) << " " << type.at(i) << ")";
//
//     con.Query(createTable.str());
//
//
//
// }


TEST_CASE("DuckDB integration test", TAG_KERNELS) {
    std::cout << "" << std::endl;
    using namespace duckdb;
    int32_t count = 100;
    int32_t mod = 10;
    int32_t seed = 0;


    DuckDB db(nullptr);
    Connection con(db);

    con.Query("CREATE TABLE integers(i INTEGER, j INTEGER)");


    Appender appender(con, "integers");
    appender.AppendRow(10000, 1001);
    appender.BeginRow();
    appender.Append(10001);
    appender.Append(1002);
    appender.EndRow();
    appender.Close();
    //Creating random data
    std::stringstream s_stream;
    s_stream << "INSERT INTO integers VALUES";
    s_stream << " (" << 0 << ", " << noise(0, mod, seed) << ")";
    for(int i = 1; i < count; i++){
      s_stream << ", (" << i << ", " << noise(i, mod, seed) << ")";
    }

    std::string insert = s_stream.str();
    //std::cout << insert << std::endl;


    con.Query(insert);
    //TODO: LOOK AT MaterializedQueryResult. the ToString results in 100 entries but it should be 1 mil..

    unique_ptr<MaterializedQueryResult> result = con.Query("SELECT * FROM integers WHERE i > 1000");
    //
    // unique_ptr<DataChunk> mainChunk = result->Fetch();
    // unique_ptr<DataChunk> nextChunk = result->Fetch();
    // //
    // // std::cout << mainChunk->data.size() << "\t" << mainChunk->data[0].GetValue(2) << std::endl;
    // // //vector<Vector> data = mainChunk->data;
    // // Vector x = move(mainChunk->data.at(0));
    // // std::cout << mainChunk->data.at(0).ToString() << std::endl;
    // std::cout << "main\t" << mainChunk->size() << "\t" << mainChunk->data[0].GetValue(0) << std::endl;
    // std::cout << "test\t" << nextChunk->size() << "\t" << nextChunk->data[0].GetValue(0) << std::endl;
    //
    // mainChunk->Append((const_cast<DataChunk&>(*nextChunk)), true);
    //
    // std::cout << "resize\t" << mainChunk->size() << "\t" << mainChunk->data[0].GetValue(0) << std::endl;
    // std::cout << "sizeof\t"<< sizeof(mainChunk->data[0].GetValue(0)) << std::endl;
    //
    // string testing1;
    // vector<Value> testing2;
    // vector<Value> testing3;
    //
    // std::cout << "\n\n"<< sizeof(testing1) << "\t" << sizeof(testing2)  << "\t" << sizeof(testing3) << "\t" << sizeof(false)  << "\n" << std::endl;
    // std::cout << sizeof(value_) << std::endl;
    std::cout << result->ToString() << std::endl;
    if (!result->success) {
        std::cerr << result->error;
    }
}
