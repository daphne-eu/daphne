/*
* Copyright 2021 The DAPHNE Consortium
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/ExternalSql.h>
#include <runtime/local/context/DaphneContext.h>
#include <tags.h>

#include <catch.hpp>
#include <string>
#include <vector>
#include <cstdint>

TEST_CASE("externalSql_duckdb_basic", TAG_KERNELS) {

   std::string query = "SELECT 42 AS a, 1234567890123 AS b, 3.14 AS c, 'hello' AS d, DATE '2025-02-03' AS e";
   const char* dbms = "DuckDB";
   const char* connection = ":memory:";  // in-memory DuckDB

   DaphneContext* ctx = nullptr;

   Frame* res = nullptr;
   externalSql(res, query.c_str(), dbms, connection, ctx);

   CHECK(res != nullptr);

   CHECK(res->getNumRows() == 1);
   CHECK(res->getNumCols() == 5);

   std::vector<std::string> expectedLabels = {"a", "b", "c", "d", "e"};
   for(size_t i = 0; i < expectedLabels.size(); ++i) {
       CHECK(res->getLabels()[i] == expectedLabels[i]);
   }

   auto expCol0 = genGivenVals<DenseMatrix<int32_t>>(1, {42});
   auto expCol1 = genGivenVals<DenseMatrix<int64_t>>(1, {1234567890123LL});
   auto expCol2 = genGivenVals<DenseMatrix<double>>(1, {3.14});
   auto expCol3 = genGivenVals<DenseMatrix<std::string>>(1, {"hello"});
   auto expCol4 = genGivenVals<DenseMatrix<std::string>>(1, {"2025-02-03"});

   CHECK(*(res->getColumn<int32_t>(0)) == *expCol0);
   CHECK(*(res->getColumn<int64_t>(1)) == *expCol1);
   CHECK(*(res->getColumn<double>(2)) == *expCol2);
   CHECK(*(res->getColumn<std::string>(3)) == *expCol3);
   CHECK(*(res->getColumn<std::string>(4)) == *expCol4);

   // Clean up
   DataObjectFactory::destroy(expCol0, expCol1, expCol2, expCol3, expCol4);
   DataObjectFactory::destroy(res);
}

// Test to check data type handling in DuckDB
TEST_CASE("externalSql_duckdb_data_type_handling", TAG_KERNELS) {
    const char* dbms = "DuckDB";
    const char* connection = ":memory:";
    DaphneContext* ctx = nullptr;

    Frame* res = nullptr;
    std::string multiQuery = "CREATE TABLE test_types (id INTEGER PRIMARY KEY, tiny_val TINYINT, small_val SMALLINT, int_val INTEGER, big_val BIGINT, bool_val BOOLEAN, str_val VARCHAR, date_val DATE); INSERT INTO test_types VALUES (1, 127, 32767, 2147483647, 9223372036854775807, TRUE, 'Test String', '2024-02-07'), (2, -128, -32768, -2147483648, -9223372036854775808, FALSE, 'Another Test', '2025-12-31'); SELECT tiny_val, small_val, int_val, big_val, bool_val, str_val, date_val FROM test_types;";

    externalSql(res, multiQuery.c_str(), dbms, connection, ctx);


    auto tinyExp = genGivenVals<DenseMatrix<int8_t>>(2, {127, -128});
    auto smallExp = genGivenVals<DenseMatrix<int32_t>>(2, {32767, -32768});
    auto intExp = genGivenVals<DenseMatrix<int32_t>>(2, {2147483647, -2147483648});
    auto bigExp = genGivenVals<DenseMatrix<int64_t>>(2, {9223372036854775807, -9223372036854775808});
    auto boolExp = genGivenVals<DenseMatrix<int8_t>>(2, {1, 0}); // TRUE → 1, FALSE → 0
    auto strExp = genGivenVals<DenseMatrix<std::string>>(2, {"Test String", "Another Test"});
    auto dateExp = genGivenVals<DenseMatrix<std::string>>(2, {"2024-02-07", "2025-12-31"});

    CHECK(*(res->getColumn<int8_t>(0)) == *tinyExp);
    CHECK(*(res->getColumn<int32_t>(1)) == *smallExp);
    CHECK(*(res->getColumn<int32_t>(2)) == *intExp);
    CHECK(*(res->getColumn<int64_t>(3)) == *bigExp);
    CHECK(*(res->getColumn<int8_t>(4)) == *boolExp);
    CHECK(*(res->getColumn<std::string>(5)) == *strExp);
    CHECK(*(res->getColumn<std::string>(6)) == *dateExp);

    DataObjectFactory::destroy(tinyExp, smallExp, intExp, bigExp, boolExp, strExp, dateExp);
    DataObjectFactory::destroy(res);
}

TEST_CASE("externalSql_duckdb_large_dataset", TAG_KERNELS) {
    std::string query = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 10000) SELECT x AS a FROM cnt";
    const char* dbms = "DuckDB";
    const char* connection = ":memory:";

    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;
    externalSql(res, query.c_str(), dbms, connection, ctx);

    CHECK(res != nullptr);
    CHECK(res->getNumRows() == 10000);
    CHECK(res->getNumCols() == 1);

    std::vector<int32_t> values(10000);
    std::iota(values.begin(), values.end(), 1); // Fills with {1, 2, ..., 1000}
    auto expCol0 = genGivenVals<DenseMatrix<int32_t>>(10000, values);

    CHECK(*(res->getColumn<int32_t>(0)) == *expCol0);

    DataObjectFactory::destroy(expCol0);
    DataObjectFactory::destroy(res);
}

// Test case using the ODBC branch (with DSN "SQLite")
/*TEST_CASE("externalSql_odbc_sqlite", TAG_KERNELS) {

    std::string query = "SELECT 42 AS a, 1234567890123 AS b, 3.14 AS c, 'hello' AS d, DATE '2025-02-03' AS e";
    // Here, we use "odbc" (or "duckdbODBC" if that is how your kernel distinguishes the branch)
    // Adjust this string to match the branch in your ExternalSql::apply.
    const char* dbms = "SQLite";
    // For the ODBC branch, the connection string is not used since DSN is hardcoded inside.
    const char* connection = "odbc";

    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;
    externalSql(res, query.c_str(), dbms, connection, ctx);

    CHECK(res != nullptr);
    CHECK(res->getNumRows() == 1);
    CHECK(res->getNumCols() == 5);

    std::vector<std::string> expectedLabels = {"a", "b", "c", "d", "e"};
    for(size_t i = 0; i < expectedLabels.size(); ++i) {
        CHECK(res->getLabels()[i] == expectedLabels[i]);
    }

    auto expCol0 = genGivenVals<DenseMatrix<int32_t>>(1, {42});
    auto expCol1 = genGivenVals<DenseMatrix<int64_t>>(1, {1234567890123LL});
    auto expCol2 = genGivenVals<DenseMatrix<double>>(1, {3.14});
    auto expCol3 = genGivenVals<DenseMatrix<std::string>>(1, {"hello"});
    auto expCol4 = genGivenVals<DenseMatrix<std::string>>(1, {"2025-02-03"});

    CHECK(*(res->getColumn<int32_t>(0)) == *expCol0);
    CHECK(*(res->getColumn<int64_t>(1)) == *expCol1);
    CHECK(*(res->getColumn<double>(2)) == *expCol2);
    CHECK(*(res->getColumn<std::string>(3)) == *expCol3);
    CHECK(*(res->getColumn<std::string>(4)) == *expCol4);

    DataObjectFactory::destroy(expCol0, expCol1, expCol2, expCol3, expCol4);
    DataObjectFactory::destroy(res);
}*/

// Test case where we expect an unsupported DBMS error
TEST_CASE("externalSql_unsupported_dbms", TAG_KERNELS) {
    std::string query = "SELECT 1";
    const char* dbms = "unsupported_dbms";
    const char* connection = "";
    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;

    // The kernel should throw an error indicating that this DBMS is unsupported.
    CHECK_THROWS_WITH(externalSql(res, query.c_str(), dbms, connection, ctx),
                      Catch::Contains("Unsupported DBMS"));
}

// Test case where we expect odbc driver not found error
TEST_CASE("externalSql_odbc_driver_not_found", TAG_KERNELS) {
    std::string query = "SELECT 1";
    const char* dbms = "unsupported_dbms";
    const char* connection = "odbc";
    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;

    // The kernel should throw an error indicating that this DBMS is unsupported.
    CHECK_THROWS_WITH(externalSql(res, query.c_str(), dbms, connection, ctx),
                      Catch::Contains("Data source name not found"));
}

// Test case where we expect an empty query error
TEST_CASE("externalSql_empty_query", TAG_KERNELS) {
    std::string query = "";
    const char* dbms = "duckdb";
    const char* connection = ":memory:";
    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;

    // The kernel should throw an error indicating that this DBMS is unsupported.
    CHECK_THROWS_WITH(externalSql(res, query.c_str(), dbms, connection, ctx),
                      Catch::Contains("Query is empty"));
}

// Test case where we expect a Parser error due to an invalid query
TEST_CASE("externalSql_invalid_query", TAG_KERNELS) {
    std::string query = "invalid";
    const char* dbms = "DuckDB";
    const char* connection = ":memory:";
    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;

    CHECK_THROWS_WITH(externalSql(res, query.c_str(), dbms, connection, ctx),
                      Catch::Contains("Parser Error"));
}

TEST_CASE("externalSql_sqlite_basic", TAG_KERNELS) {

    std::string query = "SELECT 42 AS a, 1234567890123 AS b, 3.14 AS c, 'hello' AS d, '2025-02-03' AS e";
    const char* dbms = "SQLite";
    const char* connection = ":memory:";  // in-memory SQLite

    DaphneContext* ctx = nullptr;

    Frame* res = nullptr;
    externalSql(res, query.c_str(), dbms, connection, ctx);

    CHECK(res != nullptr);

    CHECK(res->getNumRows() == 1);
    CHECK(res->getNumCols() == 5);

    std::vector<std::string> expectedLabels = {"a", "b", "c", "d", "e"};
    for(size_t i = 0; i < expectedLabels.size(); ++i) {
        CHECK(res->getLabels()[i] == expectedLabels[i]);
    }

    auto expCol0 = genGivenVals<DenseMatrix<int64_t>>(1, {42});
    auto expCol1 = genGivenVals<DenseMatrix<int64_t>>(1, {1234567890123LL});
    auto expCol2 = genGivenVals<DenseMatrix<double>>(1, {3.14});
    auto expCol3 = genGivenVals<DenseMatrix<std::string>>(1, {"hello"});
    auto expCol4 = genGivenVals<DenseMatrix<std::string>>(1, {"2025-02-03"});

    CHECK(*(res->getColumn<int64_t>(0)) == *expCol0);
    CHECK(*(res->getColumn<int64_t>(1)) == *expCol1);
    CHECK(*(res->getColumn<double>(2)) == *expCol2);
    CHECK(*(res->getColumn<std::string>(3)) == *expCol3);
    CHECK(*(res->getColumn<std::string>(4)) == *expCol4);

    // Clean up
    DataObjectFactory::destroy(expCol0, expCol1, expCol2, expCol3, expCol4);
    DataObjectFactory::destroy(res);
}

TEST_CASE("externalSql_sqlite_null_values", TAG_KERNELS) {
    std::string query = "SELECT NULL AS a, 1 AS b, 2 AS c, 3 AS d, 4 AS e";
    const char* dbms = "SQLite";
    const char* connection = ":memory:";

    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;

    CHECK_THROWS_WITH(externalSql(res, query.c_str(), dbms, connection, ctx),
                      Catch::Contains("Unsupported Type"));
}

TEST_CASE("externalSql_sqlite_large_dataset", TAG_KERNELS) {
    std::string query = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 1000000) SELECT x AS a FROM cnt";
    const char* dbms = "SQLite";
    const char* connection = ":memory:";

    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;
    externalSql(res, query.c_str(), dbms, connection, ctx);

    CHECK(res != nullptr);
    CHECK(res->getNumRows() == 1000000);
    CHECK(res->getNumCols() == 1);

    std::vector<int64_t> values(1000000);
    std::iota(values.begin(), values.end(), 1); // Fills with {1, 2, ..., 1000}
    auto expCol0 = genGivenVals<DenseMatrix<int64_t>>(1000000, values);

    CHECK(*(res->getColumn<int64_t>(0)) == *expCol0);

    DataObjectFactory::destroy(expCol0);
    DataObjectFactory::destroy(res);
}

TEST_CASE("externalSql_sqlite_multiple_columns", TAG_KERNELS) {
    std::string query = "SELECT 1 AS a, 2.5 AS b, 'test' AS c, X'01FF' AS d";
    const char* dbms = "SQLite";
    const char* connection = ":memory:";

    DaphneContext* ctx = nullptr;
    Frame* res = nullptr;
    externalSql(res, query.c_str(), dbms, connection, ctx);

    CHECK(res != nullptr);
    CHECK(res->getNumRows() == 1);
    CHECK(res->getNumCols() == 4);

    auto expCol0 = genGivenVals<DenseMatrix<int64_t>>(1, {1});
    auto expCol1 = genGivenVals<DenseMatrix<double>>(1, {2.5});
    auto expCol2 = genGivenVals<DenseMatrix<std::string>>(1, {"test"});
    auto expCol3 = genGivenVals<DenseMatrix<std::string>>(1, {std::string("\x01\xFF", 2)});

    CHECK(*(res->getColumn<int64_t>(0)) == *expCol0);
    CHECK(*(res->getColumn<double>(1)) == *expCol1);
    CHECK(*(res->getColumn<std::string>(2)) == *expCol2);
    CHECK(*(res->getColumn<std::string>(3)) == *expCol3);

    DataObjectFactory::destroy(expCol0, expCol1, expCol2, expCol3);
    DataObjectFactory::destroy(res);
}