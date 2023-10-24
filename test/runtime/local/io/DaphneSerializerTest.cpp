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
#include <runtime/local/kernels/RandMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>

#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, float, double
#define SERIALIZATION_BUFFER_SIZE 200, 1000, 1048576 // bytes

// ----------------------------------------------------------------------------
// Small defined matrices
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
    using DT = TestType;
    DT* mat = nullptr;
    if (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(5, {0, 23, 4, 94, 53,
                                    6, 13, 89, 31, 21,
                                    42, 45, 78, 35, 25,
                                    2, 23, 88, 123, 5,
                                    44, 77, 2, 1, 2});
    } else if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(5, {0, 0, 0, 0, 53,
                                    0, 0, 0, 0, 0,
                                    0, 0, 78, 0, 0,
                                    0, 0, 0, 123, 0,
                                    0, 77, 0, 0, 0});
    }

    // Serialize and deserialize
    std::vector<char> buffer;
    DaphneSerializer<DT>::serialize(mat, buffer);

    auto newMat = dynamic_cast<DT *>(DF_deserialize(buffer));

    CHECK(*newMat == *mat);

    DataObjectFactory::destroy(mat);
    DataObjectFactory::destroy(newMat);
}

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize in chunks out of order", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
    using DT = TestType;
    DT* mat = nullptr;
    if (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(10, {66, 58, 24, 118, 51, 22, 75, 32, 17, 8,
                                     74, 55, 44, 63, 51, 44, 75, 87, 63, 42,
                                     71, 108, 10, 101, 92, 34, 101, 89, 39, 91,
                                     48, 36, 69, 63, 69, 18, 7, 56, 63, 28,
                                     61, 16, 9, 87, 25, 40, 12, 27, 22, 18,
                                     11, 4, 22, 71, 94, 82, 65, 93, 45, 24,
                                     38, 93, 102, 99, 29, 90, 84, 72, 93, 10,
                                     80, 98, 18, 21, 89, 104, 12, 82, 25, 38,
                                     39, 74, 64, 26, 55, 78, 104, 93, 34, 76,
                                     123, 65, 36, 87, 48, 87, 53, 73, 31, 82});
    } else if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(10, {0, 0, 55, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 40, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 98, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 93, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }

    // Serialize in chunks and copy them to an array of messages
    size_t chunkSize = 200;
    std::vector<std::vector<char>> message;

    DaphneSerializerOutOfOrderChunks<DT> serializer(mat, chunkSize);
    // Serialize matrix
    while (serializer.HasNextChunk())
    {
        std::vector<char> bufferTmp(200);
        serializer.SerializeNextChunk(bufferTmp);
        // Push to vector of messages
        message.push_back(bufferTmp);
    }

    // Shuffle vector of messages
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(message), std::end(message), rng);

    // Deserialize
    DaphneDeserializerOutOfOrderChunks<DT> deserializer;
    size_t i = 0;
    DT *res = nullptr;
    // We want to iterate using HasNextChunk() method
    while (deserializer.HasNextChunk())
    {
        // Deserialize message
        res = deserializer.DeserializeNextChunk(message[i++]);
    }

    CHECK(*res == *mat);

    DataObjectFactory::destroy(mat);
    if (res != nullptr) // suppress warning
        DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize in order using iterator", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
    using DT = TestType;
    DT* mat = nullptr;
    if (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(10, {66, 58, 24, 118, 51, 22, 75, 32, 17, 8,
                                     74, 55, 44, 63, 51, 44, 75, 87, 63, 42,
                                     71, 108, 10, 101, 92, 34, 101, 89, 39, 91,
                                     48, 36, 69, 63, 69, 18, 7, 56, 63, 28,
                                     61, 16, 9, 87, 25, 40, 12, 27, 22, 18,
                                     11, 4, 22, 71, 94, 82, 65, 93, 45, 24,
                                     38, 93, 102, 99, 29, 90, 84, 72, 93, 10,
                                     80, 98, 18, 21, 89, 104, 12, 82, 25, 38,
                                     39, 74, 64, 26, 55, 78, 104, 93, 34, 76,
                                     123, 65, 36, 87, 48, 87, 53, 73, 31, 82});
    } else if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value) {
        mat = genGivenVals<DT>(10, {0, 0, 55, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 40, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 98, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 93, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }

    auto tempBuff = std::vector<char>(DaphneSerializer<DT>::length(mat));

    auto ser = DaphneSerializerChunks<DT>(mat, 200);

    size_t idx = 0;
    for (auto it = ser.begin(); it != ser.end(); ++it)
    {
        std::copy(it->second->begin(), it->second->begin() + it->first, tempBuff.begin() + idx);
        idx += it->first;
    }

    DT *newMat = nullptr;
    idx = 0;
    auto deser = DaphneDeserializerChunks<DT>(&newMat, 200);
    for (auto it = deser.begin(); it != deser.end(); ++it)
    {
        size_t chnck = std::min(size_t(200), tempBuff.capacity() - idx);
        std::copy(tempBuff.begin() + idx, tempBuff.begin() + idx + chnck, it->second->begin());
        it->first = chnck;
        idx += chnck;
    }
    CHECK(*newMat == *mat);

    DataObjectFactory::destroy(mat);
    if (newMat != nullptr) // suppress warning
        DataObjectFactory::destroy(newMat);
}

// ----------------------------------------------------------------------------
// Large random matrices
// ----------------------------------------------------------------------------
TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize, large random input", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
    using DT = TestType;
    const size_t numRows = 10000;
    const size_t numCols = 10000;

    // Set sparsity to 0.05 if we test for Sparse Matrix else 1.0
    double sparsity = 1.0;
    if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value){
        sparsity = 0.05;
    }
    DT * mat = nullptr;
    randMatrix<DT, typename DT::VT>(
            mat, numRows, numCols, 0, 127, sparsity, -1, nullptr
    );

    // Serialize and deserialize
    std::vector<char> buffer;
    DaphneSerializer<DT>::serialize(mat, buffer);

    auto newMat = dynamic_cast<DT *>(DF_deserialize(buffer));

    CHECK(*newMat == *mat);

    DataObjectFactory::destroy(mat);
    DataObjectFactory::destroy(newMat);
}

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize in chunks out of order, large random input", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
    using DT = TestType;
    const size_t numRows = 10000;
    const size_t numCols = 10000;
    size_t bufferSize = 524288; // 500Kb

    // Set sparsity to 0.05 if we test for Sparse Matrix else 1.0
    double sparsity = 1.0;
    if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value){
        sparsity = 0.05;
    }
    DT * mat = nullptr;
    randMatrix<DT, typename DT::VT>(
            mat, numRows, numCols, 0, 127, sparsity, -1, nullptr
    );

    // Serialize in chunks and copy them to an array of messages
    std::vector<std::vector<char>> message;

    DaphneSerializerOutOfOrderChunks<DT> serializer(mat, bufferSize);
    // Serialize matrix
    while (serializer.HasNextChunk())
    {
        std::vector<char> bufferTmp(bufferSize);
        serializer.SerializeNextChunk(bufferTmp);
        // Push to vector of messages
        message.push_back(bufferTmp);
    }

    // Shuffle vector of messages
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(message), std::end(message), rng);

    // Deserialize
    DaphneDeserializerOutOfOrderChunks<DT> deserializer;
    size_t i = 0;
    DT *res = nullptr;
    // We want to iterate using HasNextChunk() method
    while (deserializer.HasNextChunk())
    {
        // Deserialize message
        res = deserializer.DeserializeNextChunk(message[i++]);
    }

    CHECK(*res == *mat);

    DataObjectFactory::destroy(mat);
    if (res != nullptr) // suppress warning
        DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize in order using iterator, large random input", TAG_IO, (CSRMatrix), (VALUE_TYPES))
{
    using DT = TestType;
    const size_t numRows = 10000;
    const size_t numCols = 10000;
    const size_t bufferSize = 524288; // 500Kb

    // Set sparsity to 0.05 if we test for Sparse Matrix else 1.0
    double sparsity = 1.0;
    if (std::is_same<DT, CSRMatrix<typename DT::VT>>::value){
        sparsity = 0.05;
    }
    DT * mat = nullptr;
    randMatrix<DT, typename DT::VT>(
            mat, numRows, numCols, 0, 127, sparsity, -1, nullptr
    );

    auto tempBuff = std::vector<char>(DaphneSerializer<DT>::length(mat));
    auto ser = DaphneSerializerChunks<DT>(mat, bufferSize);

    size_t idx = 0;
    for (auto it = ser.begin(); it != ser.end(); ++it)
    {
        std::copy(it->second->begin(), it->second->begin() + it->first, tempBuff.begin() + idx);
        idx += it->first;
    }

    DT *newMat = nullptr;
    idx = 0;
    auto deser = DaphneDeserializerChunks<DT>(&newMat, bufferSize);
    for (auto it = deser.begin(); it != deser.end(); ++it)
    {
        size_t chnck = std::min(size_t(bufferSize), tempBuff.capacity() - idx);
        std::copy(tempBuff.begin() + idx, tempBuff.begin() + idx + chnck, it->second->begin());
        it->first = chnck;
        idx += chnck;
    }
    CHECK(*newMat == *mat);
    
    DataObjectFactory::destroy(mat);
    if (newMat != nullptr) // suppress warning
        DataObjectFactory::destroy(newMat);
}
