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
#include <runtime/local/kernels/SetColLabels.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#include <cstddef>

TEST_CASE("SetColLabels", TAG_KERNELS) {
    const size_t numCols = 3;
    
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::SI32, ValueTypeCode::UI8};
    auto f = DataObjectFactory::create<Frame>(4, numCols, schema, nullptr, false);
    
    const char * labelsIn[numCols] = {"ab", "cde", "fghi"};
    setColLabels(f, labelsIn, numCols);
    
    const std::string * labelsOut = f->getLabels();
    for(size_t i = 0; i < numCols; i++)
        CHECK(labelsOut[i] == labelsIn[i]);
    
    DataObjectFactory::destroy(f);
}