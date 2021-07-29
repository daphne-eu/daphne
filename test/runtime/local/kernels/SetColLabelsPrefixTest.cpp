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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/SetColLabelsPrefix.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#include <cstddef>

TEST_CASE("SetColLabelsPrefix", TAG_KERNELS) {
    const size_t numCols = 3;
    
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::SI32, ValueTypeCode::UI8};
    const std::string labelsIn[numCols] = {"a", "b", "c"};
    auto f = DataObjectFactory::create<Frame>(4, numCols, schema, labelsIn, false);
    
    const std::string * labelsOut;
    
    // Introduces a prefix.
    setColLabelsPrefix(f, "R", nullptr);
    
    labelsOut = f->getLabels();
    CHECK(labelsOut[0] == "R.a");
    CHECK(labelsOut[1] == "R.b");
    CHECK(labelsOut[2] == "R.c");
    
    // Changes an existing prefix.
    setColLabelsPrefix(f, "S", nullptr);
    
    labelsOut = f->getLabels();
    CHECK(labelsOut[0] == "S.a");
    CHECK(labelsOut[1] == "S.b");
    CHECK(labelsOut[2] == "S.c");
    
    DataObjectFactory::destroy(f);
}