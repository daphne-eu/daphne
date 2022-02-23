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

TEMPLATE_PRODUCT_TEST_CASE("ReadMM", TAG_KERNELS, (DenseMatrix), (double)) {
  using DT = TestType;
  DT *m = nullptr;

  size_t numRows = 497;
  size_t numCols = 507;

  char filename[] = "./test/runtime/local/io/crg.mtx";
  readMM(m, filename);

  REQUIRE(m->getNumRows() == numRows);
  REQUIRE(m->getNumCols() == numCols);

  CHECK(m->get(5, 0) == 0.25599762);

  DataObjectFactory::destroy(m);
}