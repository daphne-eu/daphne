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
#include <runtime/local/kernels/Agg.h>
#include <runtime/local/kernels/AggSparse.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

TEST_CASE("Agg-ContiguousTensor", TAG_KERNELS) {
    auto t1   = DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({3, 3, 3}), InitCode::IOTA);
    auto tmod = DataObjectFactory::create<ContiguousTensor<uint64_t>>(std::vector<size_t>({3, 3, 2}), InitCode::IOTA);
    tmod->data[0] = 1;

    bool all_true[3] = {true,true,true};

    ContiguousTensor<uint32_t>* t2 = agg<ContiguousTensor<uint32_t>, ContiguousTensor<uint32_t>>(
      AggOpCode::SUM, all_true, t1, nullptr);
    ContiguousTensor<uint64_t>* t3 = agg<ContiguousTensor<uint64_t>, ContiguousTensor<uint64_t>>(
      AggOpCode::PROD, all_true, tmod, nullptr);
    ContiguousTensor<uint32_t>* t4 = agg<ContiguousTensor<uint32_t>, ContiguousTensor<uint32_t>>(
      AggOpCode::MIN, all_true, t1, nullptr);
    ContiguousTensor<uint32_t>* t5 = agg<ContiguousTensor<uint32_t>, ContiguousTensor<uint32_t>>(
      AggOpCode::MAX, all_true, t1, nullptr);

    REQUIRE((t2->tensor_shape == std::vector<size_t>({1, 1, 1})));
    REQUIRE(t2->data[0] == 351);
    REQUIRE(t3->data[0] == 355687428096000);
    REQUIRE(t4->data[0] == 0);
    REQUIRE(t5->data[0] == 26);

    auto tc = DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({3, 3, 3}), InitCode::NONE);
    for (size_t i = 0; i < 27; i++) {
        tc->data[i] = 1;
    }

    bool b1[3] = {false,true,true};

    ContiguousTensor<uint32_t>* tc1 = agg<ContiguousTensor<uint32_t>, ContiguousTensor<uint32_t>>(
      AggOpCode::SUM, b1, tc, nullptr);
    REQUIRE(tc1->tensor_shape == std::vector<size_t>({3, 1, 1}));
    for (size_t i = 0; i < 3; i++) {
        REQUIRE(tc1->get({i, 0, 0}) == 9);
    }
}

TEST_CASE("Agg-ChunkedTensor", TAG_KERNELS) {
    auto t = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({4, 4}), std::vector<size_t>({4, 2}), InitCode::IOTA);
    auto t1 = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({4, 4, 2}), std::vector<size_t>({2, 1, 2}), InitCode::IOTA);
    auto tmod = DataObjectFactory::create<ChunkedTensor<uint64_t>>(
      std::vector<size_t>({2, 4, 2}), std::vector<size_t>({2, 1, 2}), InitCode::IOTA);
    tmod->data[0] = 1;

    bool b5[2] = {true, true};
    // Very simple just 2D and 2 chunks
    ChunkedTensor<uint32_t>* ta = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, b5, t, nullptr);
    REQUIRE((ta->tensor_shape == std::vector<size_t>({1, 1})));
    REQUIRE((ta->chunk_shape == std::vector<size_t>({1, 1})));
    REQUIRE(ta->data[0] == 120);

    bool all_true[3] = {true,true,true};

    // 3D still reduce all dims, different OPs
    ChunkedTensor<uint32_t>* t2 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, all_true, t1, nullptr);
    ChunkedTensor<uint64_t>* t3 = agg<ChunkedTensor<uint64_t>, ChunkedTensor<uint64_t>>(
      AggOpCode::PROD, all_true, tmod, nullptr);
    ChunkedTensor<uint32_t>* t4 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::MIN, all_true, t1, nullptr);
    ChunkedTensor<uint32_t>* t5 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::MAX, all_true, t1, nullptr);

    REQUIRE((t2->tensor_shape == std::vector<size_t>({1, 1, 1})));
    REQUIRE((t2->chunk_shape == std::vector<size_t>({1, 1, 1})));
    REQUIRE(t2->data[0] == 496);
    REQUIRE(t3->data[0] == 1307674368000);
    REQUIRE(t4->data[0] == 0);
    REQUIRE(t5->data[0] == 31);

    // 3D reduce 2 dims
    auto tc = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({4, 4, 2}), std::vector<size_t>({2, 1, 2}), InitCode::IOTA);
    for (size_t i = 0; i < 32; i++) {
        tc->data[i] = 1;
    }

    bool b1[3] = {false,true,true};
    bool b2[3] = {true,false,true};
    bool b3[3] = {false,false,false};

    ChunkedTensor<uint32_t>* tc1 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, b1, tc, nullptr);
    ChunkedTensor<uint32_t>* tc2 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, b2, tc, nullptr);

    REQUIRE(tc1->tensor_shape == std::vector<size_t>({4, 1, 1}));
    REQUIRE(tc1->chunk_shape == std::vector<size_t>({2, 1, 1}));
    REQUIRE(tc2->tensor_shape == std::vector<size_t>({1, 4, 1}));
    REQUIRE(tc2->chunk_shape == std::vector<size_t>({1, 1, 1}));

    for (size_t i = 0; i < 4; i++) {
        REQUIRE(tc1->get({i, 0, 0}) == 8);
        REQUIRE(tc2->get({0, i, 0}) == 8);
    }

    // Reduce no dim i.e. NOOP
    ChunkedTensor<uint32_t>* tc3 = agg<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, b3, tc, nullptr);

    REQUIRE(tc3->tensor_shape == std::vector<size_t>({4, 4, 2}));
    REQUIRE(tc3->chunk_shape == std::vector<size_t>({2, 1, 2}));
    for (size_t i = 0; i < 32; i++) {
        REQUIRE(tc3->data[i] == tc->data[i]);
    }
}

TEST_CASE("Agg-ChunkedTensor-Sparse", TAG_KERNELS) {
    auto t = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({4, 4, 4}), std::vector<size_t>({4, 2, 2}), InitCode::IOTA);

    bool all_true[3] = {true,true,true};
    size_t lhs_range_values[3] = {0,0,0};
    size_t rhs_range_values[3] = {1,2,1};
    size_t rhs_range_values1[3] = {1,1,1};
    size_t rhs_range_values2[3] = {1,1,4};

    // Agg over all dims but only supply [0,1) interval in last dim
    ChunkedTensor<uint32_t>* tr1 = aggSparse<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, all_true, lhs_range_values, rhs_range_values, t, nullptr);

    REQUIRE(tr1->tensor_shape == std::vector<size_t>({1, 1, 1}));
    REQUIRE(tr1->chunk_shape == std::vector<size_t>({1, 1, 1}));

    REQUIRE(tr1->data[0] == 496);

    ChunkedTensor<uint32_t>* tr2 = aggSparse<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, all_true, lhs_range_values, rhs_range_values1, t, nullptr);

    REQUIRE(tr2->tensor_shape == std::vector<size_t>({1, 1, 1}));
    REQUIRE(tr2->chunk_shape == std::vector<size_t>({1, 1, 1}));

    size_t expected = 0;
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                expected += t->get({i, j, k});
            }
        }
    }
    REQUIRE(tr2->data[0] == expected);

    bool b1[3] = {true,true,false};

    auto t2 = DataObjectFactory::create<ChunkedTensor<uint32_t>>(
      std::vector<size_t>({4, 4, 4}), std::vector<size_t>({4, 2, 1}), InitCode::IOTA);

    ChunkedTensor<uint32_t>* t2r1 = aggSparse<ChunkedTensor<uint32_t>, ChunkedTensor<uint32_t>>(
      AggOpCode::SUM, b1, lhs_range_values, rhs_range_values2, t2, nullptr);

    REQUIRE(t2r1->tensor_shape == std::vector<size_t>({1, 1, 4}));
    REQUIRE(t2r1->chunk_shape == std::vector<size_t>({1, 1, 1}));

    size_t exp[4] = {0,0,0,0};
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 2; j++) {
          for(size_t k=0; k<4; k++) {
            exp[k] += t->get({i, j, k});
          }
        }
    }
    REQUIRE(t2r1->data[0] == exp[0]);
    REQUIRE(t2r1->data[1] == exp[1]);
    REQUIRE(t2r1->data[2] == exp[2]);
    REQUIRE(t2r1->data[3] == exp[3]);
}
