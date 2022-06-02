/*
 * Copyright 2022 The DAPHNE Consortium
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
#include <runtime/local/io/ReadMM.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <limits>

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CIG", TAG_IO, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 9;
  size_t numCols = 9;

  char filename[] = "./test/runtime/local/io/cig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(2, 0) == 0);
  CHECK(m->get(3, 4) == 9);
  CHECK(m->get(7, 4) == 4);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIG", TAG_IO, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/aig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(1, 0) == 2);
  CHECK(m->get(0, 1) == 5);
  CHECK(m->get(3, 2) == 12);
  CHECK(m->get(2, 1) == 7);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRG", TAG_IO, (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 497;
  size_t numCols = 507;

  char filename[] = "./test/runtime/local/io/crg.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(5, 0) == 0.25599762);
  CHECK(m->get(6, 0) == 0.13827993);
  CHECK(m->get(200, 4) == 0.20001954);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRS", TAG_IO, (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crs.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(36, 29) == 926.188986068);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == m->get(c,r));

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRK", TAG_IO, (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crk.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(29, 36) == -926.188986068);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->get(r,r) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == -m->get(c,r));
  }

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CPS", TAG_IO, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 24;
  size_t numCols = 24;

  char filename[] = "./test/runtime/local/io/cps.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get( 0, 0) != 0);
  CHECK(m->get( 1, 0) == 0);
  CHECK(m->get(3, 15) != 0);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      if(m->get(r,c) == 0)
        CHECK(m->get(c,r) == 0);
      else
        CHECK(m->get(c,r) != 0);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIK", TAG_IO, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/aik.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(1, 0) == 1);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->get(r,r) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == -m->get(c,r));
  }

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIS", TAG_IO, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 3;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/ais.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(1, 1) == 4);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == m->get(c,r));

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CIG (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 9;
  size_t numCols = 9;

  char filename[] = "./test/runtime/local/io/cig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(2, 0) == 0);
  CHECK(m->get(3, 4) == 9);
  CHECK(m->get(7, 4) == 4);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIG (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/aig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(1, 0) == 2);
  CHECK(m->get(0, 1) == 5);
  CHECK(m->get(3, 2) == 12);
  CHECK(m->get(2, 1) == 7);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRG (CSR)", TAG_IO, (CSRMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 497;
  size_t numCols = 507;

  char filename[] = "./test/runtime/local/io/crg.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(5, 0) == 0.25599762);
  CHECK(m->get(6, 0) == 0.13827993);
  CHECK(m->get(200, 4) == 0.20001954);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRS (CSR)", TAG_IO, (CSRMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crs.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(36, 29) == 926.188986068);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == m->get(c,r));

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRK (CSR)", TAG_IO, (CSRMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crk.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(29, 36) == -926.188986068);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->get(r,r) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == -m->get(c,r));
  }

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CPS (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 24;
  size_t numCols = 24;

  char filename[] = "./test/runtime/local/io/cps.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get( 0, 0) != 0);
  CHECK(m->get( 1, 0) == 0);
  CHECK(m->get(3, 15) != 0);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      if(m->get(r,c) == 0)
        CHECK(m->get(c,r) == 0);
      else
        CHECK(m->get(c,r) != 0);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIK (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/aik.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(1, 0) == 1);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->get(r,r) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == -m->get(c,r));
  }

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIS (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 3;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/ais.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(1, 1) == 4);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->get(r,c) == m->get(c,r));

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CIG (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 9;
  size_t numCols = 9;

  char filename[] = "./test/runtime/local/io/cig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<int64_t>(0)->get(0, 0) == 1);
  CHECK(m->getColumn<int64_t>(0)->get(2, 0) == 0);
  CHECK(m->getColumn<int64_t>(4)->get(3, 0) == 9);
  CHECK(m->getColumn<int64_t>(4)->get(7, 0) == 4);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIG (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/aig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<int64_t>(0)->get(0, 0) == 1);
  CHECK(m->getColumn<int64_t>(0)->get(1, 0) == 2);
  CHECK(m->getColumn<int64_t>(1)->get(0, 0) == 5);
  CHECK(m->getColumn<int64_t>(2)->get(3, 0) == 12);
  CHECK(m->getColumn<int64_t>(1)->get(2, 0) == 7);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRG (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 497;
  size_t numCols = 507;

  char filename[] = "./test/runtime/local/io/crg.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(0)->get(5, 0) == 0.25599762);
  CHECK(m->getColumn<double>(0)->get(6, 0) == 0.13827993);
  CHECK(m->getColumn<double>(4)->get(200, 0) == 0.20001954);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRS (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crs.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(29)->get(36, 0) == 926.188986068);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->getColumn<double>(c)->get(r,0)
        ==  m->getColumn<double>(r)->get(c,0));

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRK (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 66;
  size_t numCols = 66;

  char filename[] = "./test/runtime/local/io/crk.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(36)->get(29, 0) == -926.188986068);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->getColumn<double>(r)->get(r,0) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->getColumn<double>(c)->get(r,0)
        == -m->getColumn<double>(r)->get(c,0));
  }

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CPS (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 24;
  size_t numCols = 24;

  char filename[] = "./test/runtime/local/io/cps.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<double>(0)->get( 0, 0) != 0);
  CHECK(m->getColumn<double>(0)->get( 1, 0) == 0);
  CHECK(m->getColumn<double>(15)->get(3, 0) != 0);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      if(m->getColumn<double>(c)->get(r,0) == 0)
        CHECK(m->getColumn<double>(r)->get(c,0) == 0);
      else
        CHECK(m->getColumn<double>(r)->get(c,0) != 0);

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIK (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 4;
  size_t numCols = 4;

  char filename[] = "./test/runtime/local/io/aik.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<int64_t>(0)->get(1, 0) == 1);

  for(size_t r = 0; r<numRows; r++) {
    CHECK(m->getColumn<int64_t>(r)->get(r,0) == 0);
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->getColumn<int64_t>(c)->get(r,0)
        == -m->getColumn<int64_t>(r)->get(c,0));
  }

  DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIS (Frame)", TAG_IO) {
  using DT = Frame;
  DT *m = nullptr;

  size_t numRows = 3;
  size_t numCols = 3;

  char filename[] = "./test/runtime/local/io/ais.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->getColumn<int64_t>(1)->get(1, 0) == 4);

  for(size_t r = 0; r<numRows; r++)
    for(size_t c = r+1; c<numCols; c++)
      CHECK(m->getColumn<int64_t>(c)->get(r,0)
        ==  m->getColumn<int64_t>(r)->get(c,0));

  DataObjectFactory::destroy(m);
}
