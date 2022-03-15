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

#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/ReadParquet.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <limits>

TEST_CASE("ReadParquet", TAG_KERNELS) {
  ValueTypeCode schema[] = { ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64 };
  Frame *m = NULL;

  size_t numRows = 2;
  size_t numCols = 4;

  const char filename[] = "./test/runtime/local/io/ReadParquet1.parquet";
  const char *fn = &filename[0];

  readParquet(m, fn, numRows, numCols, schema);

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
