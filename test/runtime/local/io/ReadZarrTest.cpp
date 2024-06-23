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

#include <cstring>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ReadZarr.h>
#include <runtime/local/io/ZarrFileMetaData.h>

#include <tags.h>

#include <catch.hpp>

TEST_CASE("ReadZarr->ContiguousTensor-small", TAG_IO) {
    ChunkedTensor<double>* golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ContiguousTensor<double>* ct_ptr;

    // Read in tensor_shape = chunk_shape = [10,10,10] fp64 tensor
    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ContiguousTensorTest/example.zarr");

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->getPtrToChunk(std::vector<size_t>({0,0,0})), golden_sample->chunk_element_count * sizeof(double)) == 0);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
}

TEST_CASE("ReadZarr->ContiguousTensor-large", TAG_IO) {
    ContiguousTensor<double>* golden_sample = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({100,100,100}), InitCode::IOTA);
    ContiguousTensor<double>* ct_ptr;

    // Read in chunk_shape = [10,10,10], tensor_shape = [100,100,100] fp64 tensor into a contiguous tensor with
    // shape [100,100,100]
    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr");

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({100,100,100}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
}

TEST_CASE("ReadZarrPartial->ContiguousTensor: alinged-single-chunk", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ContiguousTensor<double>* golden_sample = DataObjectFactory::create<ContiguousTensor<double>>(full_golden_sample->getPtrToChunk(std::vector<size_t>({1,0,0})), std::vector<size_t>({10,10,10}));
    ContiguousTensor<double>* ct_ptr;

    // Read in chunk_shape = [10,10,10], tensor_shape = [100,100,100] fp64 tensor with range {[10,20),[0,10),[0,10)}
    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{10,20},{0,10},{0,10}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}

TEST_CASE("ReadZarrPartial->ContiguousTensor: alinged-multiple-chunk", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ContiguousTensor<double>* golden_sample = full_golden_sample->tryDiceToContiguousTensor({{0,20},{0,10},{0,10}});
    REQUIRE(golden_sample != nullptr);
    ContiguousTensor<double>* ct_ptr;

    // Read in chunk_shape = [10,10,10], tensor_shape = [100,100,100] fp64 tensor with range {[0,20),[0,10),[0,10)}
    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{0,20},{0,10},{0,10}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({20,10,10}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}

TEST_CASE("ReadZarrPartial->ContiguousTensor: non-alinged", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ContiguousTensor<double>* golden_sample = full_golden_sample->tryDiceToContiguousTensor({{2,11},{1,7},{2,4}});
    REQUIRE(golden_sample != nullptr);
    ContiguousTensor<double>* ct_ptr;

    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{2,11},{1,7},{2,4}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({9,6,2}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}

TEST_CASE("ReadZarr->ChunkedTensor: large", TAG_IO) {
    ChunkedTensor<double>* golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ChunkedTensor<double>* ct_ptr;

    readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr");

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({100,100,100}));
    REQUIRE(ct_ptr->chunk_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
}

TEST_CASE("ReadZarrPartial->ChunkedTensor: alinged", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ChunkedTensor<double>* golden_sample = full_golden_sample->tryDiceAtChunkLvl({{0,2},{0,1},{0,1}});
    REQUIRE(golden_sample != nullptr);
    REQUIRE(golden_sample->tensor_shape == std::vector<size_t>({20,10,10}));
    REQUIRE(golden_sample->chunk_shape == std::vector<size_t>({10,10,10}));
    ChunkedTensor<double>* ct_ptr;

    readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{0,20},{0,10},{0,10}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({20,10,10}));
    REQUIRE(ct_ptr->chunk_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(std::memcmp(ct_ptr->data.get(), golden_sample->data.get(), golden_sample->total_element_count * sizeof(double)) == 0);

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}

TEST_CASE("ReadZarrPartial->ChunkedTensor: right overhanging chunks", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ChunkedTensor<double>* golden_sample = full_golden_sample->tryDice({{0,16},{0,10},{0,10}}, full_golden_sample->chunk_shape);
    REQUIRE(golden_sample != nullptr);
    REQUIRE(golden_sample->tensor_shape == std::vector<size_t>({16,10,10}));
    REQUIRE(golden_sample->chunk_shape == std::vector<size_t>({10,10,10}));
    ChunkedTensor<double>* ct_ptr;

    readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{0,16},{0,10},{0,10}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({16,10,10}));
    REQUIRE(ct_ptr->chunk_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}

TEST_CASE("ReadZarrPartial->ChunkedTensor: both non alinged", TAG_IO) {
    ChunkedTensor<double>* full_golden_sample = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({100,100,100}),std::vector<size_t>({10,10,10}), InitCode::IOTA);
    ChunkedTensor<double>* golden_sample = full_golden_sample->tryDice({{1,3},{2,4},{11,18}}, full_golden_sample->chunk_shape);
    REQUIRE(golden_sample != nullptr);
    REQUIRE(golden_sample->tensor_shape == std::vector<size_t>({2,2,7}));
    REQUIRE(golden_sample->chunk_shape == std::vector<size_t>({10,10,10}));
    ChunkedTensor<double>* ct_ptr;

    readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{1,3},{2,4},{11,18}});

    REQUIRE(ct_ptr->tensor_shape == std::vector<size_t>({2,2,7}));
    REQUIRE(ct_ptr->chunk_shape == std::vector<size_t>({10,10,10}));

    REQUIRE(*ct_ptr == *golden_sample);

    DataObjectFactory::destroy(ct_ptr);
    DataObjectFactory::destroy(golden_sample);
    DataObjectFactory::destroy(full_golden_sample);
}
