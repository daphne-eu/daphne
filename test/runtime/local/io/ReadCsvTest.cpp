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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/File.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <limits>

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv", TAG_IO, (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv1.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == -0.1);
  CHECK(m->get(0, 1) == -0.2);
  CHECK(m->get(0, 2) == 0.1);
  CHECK(m->get(0, 3) == 0.2);

  CHECK(m->get(1, 0) == 3.14);
  CHECK(m->get(1, 1) == 5.41);
  CHECK(m->get(1, 2) == 6.22216);
  CHECK(m->get(1, 3) == 5);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv", TAG_IO, (DenseMatrix), (uint8_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(0, 1) == 2);
  CHECK(m->get(0, 2) == 3);
  CHECK(m->get(0, 3) == 4);

  /* File contains negative numbers. Expect cast to positive */
  CHECK(m->get(1, 0) == 255);
  CHECK(m->get(1, 1) == 254);
  CHECK(m->get(1, 2) == 253);
  CHECK(m->get(1, 3) == 252);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv, col + row ignore", TAG_IO,
                           (DenseMatrix), (int8_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 1;
  size_t numCols = 2;

  char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(0, 1) == 2);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv, INF and NAN parsing", TAG_IO,
                           (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv3.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == -std::numeric_limits<double>::infinity());
  CHECK(m->get(0, 1) == std::numeric_limits<double>::infinity());
  CHECK(m->get(0, 2) == -std::numeric_limits<double>::infinity());
  CHECK(m->get(0, 3) == std::numeric_limits<double>::infinity());

  CHECK(std::isnan(m->get(1, 0)));
  CHECK(std::isnan(m->get(1, 1)));
  CHECK(std::isnan(m->get(1, 2)));
  CHECK(std::isnan(m->get(1, 3)));

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, frame of floats", TAG_IO) {
  ValueTypeCode schema[] = { ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64 };
  Frame *m = NULL;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv1.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim, schema);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(0)->get(0, 0) == -0.1);
  CHECK(m->getColumn<double>(1)->get(0, 0) == -0.2);
  CHECK(m->getColumn<double>(2)->get(0, 0) == 0.1);
  CHECK(m->getColumn<double>(3)->get(0, 0) == 0.2);

  CHECK(m->getColumn<double>(0)->get(1, 0) == 3.14);
  CHECK(m->getColumn<double>(1)->get(1, 0) == 5.41);
  CHECK(m->getColumn<double>(2)->get(1, 0) == 6.22216);
  CHECK(m->getColumn<double>(3)->get(1, 0) == 5);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, frame of uint8s", TAG_IO) {
  ValueTypeCode schema[] = { ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8 };
  Frame *m = NULL;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim, schema);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
  CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);
  CHECK(m->getColumn<uint8_t>(2)->get(0, 0) == 3);
  CHECK(m->getColumn<uint8_t>(3)->get(0, 0) == 4);

  /* File contains negative numbers. Expect cast to positive */
  CHECK(m->getColumn<uint8_t>(0)->get(1, 0) == 255);
  CHECK(m->getColumn<uint8_t>(1)->get(1, 0) == 254);
  CHECK(m->getColumn<uint8_t>(2)->get(1, 0) == 253);
  CHECK(m->getColumn<uint8_t>(3)->get(1, 0) == 252);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, col + row ignore", TAG_IO) {
  ValueTypeCode schema[] = { ValueTypeCode::UI8, ValueTypeCode::UI8 };
  Frame *m = NULL;

  size_t numRows = 1;
  size_t numCols = 2;

  char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim, schema);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
  CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, INF and NAN parsing", TAG_IO) {
  ValueTypeCode schema[] = { ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64 };
  Frame *m = NULL;

  size_t numRows = 2;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/ReadCsv3.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim, schema);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(0)->get(0, 0) == -std::numeric_limits<double>::infinity());
  CHECK(m->getColumn<double>(1)->get(0, 0) == std::numeric_limits<double>::infinity());
  CHECK(m->getColumn<double>(2)->get(0, 0) == -std::numeric_limits<double>::infinity());
  CHECK(m->getColumn<double>(3)->get(0, 0) == std::numeric_limits<double>::infinity());

  CHECK(std::isnan(m->getColumn<double>(0)->get(1, 0)));
  CHECK(std::isnan(m->getColumn<double>(1)->get(1, 0)));
  CHECK(std::isnan(m->getColumn<double>(2)->get(1, 0)));
  CHECK(std::isnan(m->getColumn<double>(3)->get(1, 0)));

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, varying columns", TAG_IO) {
  ValueTypeCode schema[] = { ValueTypeCode::SI8, ValueTypeCode::F32 };
  Frame *m = NULL;

  size_t numRows = 2;
  size_t numCols = 2;

  char filename[] = "./test/runtime/local/io/ReadCsv4.csv";
  char delim = ',';

  readCsv(m, filename, numRows, numCols, delim, schema);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<int8_t>(0)->get(0, 0) == 1);
  CHECK(m->getColumn<float>(1)->get(0, 0) == 0.5);

  CHECK(m->getColumn<int8_t>(0)->get(1, 0) == 2);
  CHECK(m->getColumn<float>(1)->get(1, 0) == 1.0);

  DataObjectFactory::destroy(m);

}
