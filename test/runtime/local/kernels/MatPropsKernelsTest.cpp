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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <tags.h>
#include <catch.hpp>

bool returnsFalse() {
    return false;
}

TEMPLATE_PRODUCT_TEST_CASE("MatPropsKernelsTest", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {

  using DT = TestType;

  auto symMat = genGivenVals<DT>(4, {
          0, 1, 2, 3,
          1, 1, 4, 6,
          2, 4, 2, 7,
          3, 6, 7, 3
  });

  auto asymMat = genGivenVals<DT>(4, {
          0, 1, 2, 3,
          0, 1, 4, 6,
          2, 4, 2, 7,
          3, 6, 7, 3
  });

  auto nonSquareMat = genGivenVals<DT>(3, {
          0, 1, 2, 3,
          0, 1, 4, 6,
          2, 4, 2, 7
  });

  SECTION("isSymmetric") {
    CHECK(returnsFalse());
  }
  SECTION("hasSpecialValue") {
    CHECK(returnsFalse());
  }
  SECTION("numDistinctApprox") {
    CHECK(returnsFalse());
  }
}
