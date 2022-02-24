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
#include <runtime/local/io/mmio.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <limits>

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CIG", TAG_KERNELS, (DenseMatrix), (int32_t)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 9;
  size_t numCols = 9;

  char filename[] = "./test/runtime/local/io/cig.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(0, 0) == 1);
  CHECK(m->get(3, 4) == 9);
  CHECK(m->get(7, 4) == 4);

  DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIG", TAG_KERNELS, (DenseMatrix), (int32_t)) {
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