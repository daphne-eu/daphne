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

#include <runtime/local/kernels/Quantize.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("Quantization", TAG_KERNELS, (DenseMatrix, Matrix), (float)) {
    using DT = TestType;
    using DTRes = typename DT::template WithValueType<uint8_t>;

    auto f0 = genGivenVals<DT>(2, {
        0,   1.0,
        0.5, 1.1
    });

    DTRes * res = nullptr;

    quantize(res, f0, 0, 1, nullptr);

    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 2);

    CHECK(res->get(0,0) == 0);
    CHECK(res->get(0,1) == 255);
    CHECK(res->get(1,0) == 128);
    CHECK(res->get(1,1) == 255);
}
