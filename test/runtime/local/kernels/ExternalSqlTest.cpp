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

TEST_CASE("externalSql_duckdb", TAG_KERNELS) {

   std::string query = "SELECT 42 AS a, 1234567890123 AS b, 3.14 AS c, 'hello' AS d, DATE '2025-02-03' AS e";
   const char* dbms = "duckdb";
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

// Test case using the ODBC branch (with DSN "DuckDB")
// In this branch, the kernel uses "DSN=DuckDB" (as hardcoded in the ODBC code).
TEST_CASE("externalSql_odbc_duckdb", TAG_KERNELS) {

    std::string query = "SELECT 42 AS a, 1234567890123 AS b, 3.14 AS c, 'hello' AS d, DATE '2025-02-03' AS e";
    // Here, we use "odbc" (or "duckdbODBC" if that is how your kernel distinguishes the branch)
    // Adjust this string to match the branch in your ExternalSql::apply.
    const char* dbms = "duckdb";
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
}

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