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
#include <runtime/local/kernels/Quantize.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <tags.h>
#include <catch.hpp>

#include <vector>
#include <cstdint>

TEST_CASE("Quantization", TAG_KERNELS) {

    auto f0 = genGivenVals<DenseMatrix<float>>(2, {
        0, 1.0,
	0.5, 1.1});

    DenseMatrix<uint8_t>* res = nullptr;

    quantize<DenseMatrix<uint8_t>,DenseMatrix<float>,float>(res, f0, 0, 1, nullptr);

    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 2);

    CHECK(res->get(0,0) == 0);
    CHECK(res->get(0,1) == 255);
    CHECK(res->get(1,0) == 128);
    CHECK(res->get(1,1) == 255);

    ContiguousTensor<double>* t1 =
        DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,2}), InitCode::NONE);
    ContiguousTensor<uint8_t>* t2 = nullptr;
    t1->data[0] = 0;
    t1->data[1] = 1.0;
    t1->data[2] = 0.5;
    t1->data[3] = 1.1;

    quantize<ContiguousTensor<uint8_t>,ContiguousTensor<double>,double>(t2, t1, 0, 1, nullptr);

    CHECK(t2->data[0] == 0);
    CHECK(t2->data[1] == 255);
    CHECK(t2->data[2] == 128);
    CHECK(t2->data[3] == 255);
}
