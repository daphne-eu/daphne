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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/kernels/CheckEq.h>


#include <tags.h>

#include <catch.hpp>

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>

#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, float, double

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer save/load", TAG_IO, (DATA_TYPES), (VALUE_TYPES)) {
  using DT = TestType;  

  auto mat = genGivenVals<DT>(5, {
        0, 23, 4, 94, 53,
        6, 13, 89, 31, 21,
        42, 45, 78, 35, 25,
        2, 23, 88, 123, 5,
        44, 77, 2, 1, 2
    });
  
  // Serialize and deserialize
  std::vector<char> buffer;
  DaphneSerializer<DT>::save(mat, buffer);

  auto newMat = dynamic_cast<DT*>(DF_load(buffer));

  CHECK(*newMat == *mat);

  DataObjectFactory::destroy(mat);
  DataObjectFactory::destroy(newMat);
}