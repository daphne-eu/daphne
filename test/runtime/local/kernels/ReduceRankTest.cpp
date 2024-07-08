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

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/Tensor.h"
#include "runtime/local/datastructures/ChunkedTensor.h"
#include "runtime/local/datastructures/ContiguousTensor.h"
#include <runtime/local/kernels/ReduceRank.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

TEST_CASE("ReduceRank Tests", TAG_KERNELS) {
    ContiguousTensor<uint32_t>* t1 =
      DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({3, 1, 3}), InitCode::IOTA);
    ContiguousTensor<uint32_t>* t2 =
      DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({1, 1, 1}), InitCode::IOTA);

    reduceRank(t1, 1, nullptr);
    REQUIRE(t1->rank == 2);
    REQUIRE(t1->tensor_shape == std::vector<size_t>({3, 3}));

    reduceRank(t2, 0, nullptr);
    REQUIRE(t2->rank == 2);
    REQUIRE(t2->tensor_shape == std::vector<size_t>({1, 1}));
    reduceRank(t2, 0, nullptr);
    REQUIRE(t2->rank == 1);
    REQUIRE(t2->tensor_shape == std::vector<size_t>({1}));
    reduceRank(t2, 0, nullptr);
    REQUIRE(t2->rank == 0);
    REQUIRE(t2->tensor_shape == std::vector<size_t>({}));

    ChunkedTensor<uint32_t>* t3 = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({3, 1, 3}), std::vector<size_t>({1, 1, 1}), InitCode::IOTA);
    ChunkedTensor<uint32_t>* t4 = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({1, 1, 1}), std::vector<size_t>({1, 1, 1}), InitCode::IOTA);

    reduceRank(t3, 1, nullptr);
    REQUIRE(t3->rank == 2);
    REQUIRE(t3->tensor_shape == std::vector<size_t>({3, 3}));

    reduceRank(t4, 0, nullptr);
    REQUIRE(t4->rank == 2);
    REQUIRE(t4->tensor_shape == std::vector<size_t>({1, 1}));
    reduceRank(t4, 0, nullptr);
    REQUIRE(t4->rank == 1);
    REQUIRE(t4->tensor_shape == std::vector<size_t>({1}));
    reduceRank(t4, 0, nullptr);
    REQUIRE(t4->rank == 0);
    REQUIRE(t4->tensor_shape == std::vector<size_t>({}));

    DataObjectFactory::destroy(t1);
    DataObjectFactory::destroy(t2);
    DataObjectFactory::destroy(t3);
    DataObjectFactory::destroy(t4);
}
