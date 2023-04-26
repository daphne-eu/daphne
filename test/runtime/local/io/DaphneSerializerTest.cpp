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

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize", TAG_IO, (DATA_TYPES), (VALUE_TYPES)) {
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
  DaphneSerializer<DT>::serialize(mat, buffer);

  auto newMat = dynamic_cast<DT*>(DF_deserialize(buffer));

  CHECK(*newMat == *mat);

  DataObjectFactory::destroy(mat);
  DataObjectFactory::destroy(newMat);
}

TEMPLATE_PRODUCT_TEST_CASE("DaphneSerializer serialize/deserialize in chunks out of order", TAG_IO, (DATA_TYPES), (VALUE_TYPES))
{
  using DT = TestType;

  auto mat = genGivenVals<DT>(10, {
      66, 58, 24, 118, 51, 22, 75, 32, 17, 8,
      74, 55, 44, 63, 51, 44, 75, 87, 63, 42,
      71, 108, 10, 101, 92, 34, 101, 89, 39, 91,
      48, 36, 69, 63, 69, 18, 7, 56, 63, 28,
      61, 16, 9, 87, 25, 40, 12, 27, 22, 18,
      11, 4, 22, 71, 94, 82, 65, 93, 45, 24,
      38, 93, 102, 99, 29, 90, 84, 72, 93, 10,
      80, 98, 18, 21, 89, 104, 12, 82, 25, 38,
      39, 74, 64, 26, 55, 78, 104, 93, 34, 76,
      123, 65, 36, 87, 48, 87, 53, 73, 31, 82
  });

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