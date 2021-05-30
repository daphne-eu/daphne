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

#include <runtime/local/kernels/CastSca.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEST_CASE("castSca, no-op casts", TAG_KERNELS) {
    CHECK(castSca<int64_t, int64_t>(123) == 123);
    CHECK(castSca<double, double>(123.4) == 123.4);
    CHECK(castSca<bool, bool>(false) == false);
    CHECK(castSca<bool, bool>(true) == true);
}

TEST_CASE("castSca, actual casts", TAG_KERNELS) {
    CHECK(castSca<int64_t, double>(123.4) == 123);
    CHECK(castSca<int64_t, double>(-123.4) == -123);
    CHECK(castSca<double, int64_t>(123) == 123.0);
    CHECK(castSca<double, int64_t>(-123) == -123.0);
    
    CHECK(castSca<int64_t, bool>(false) == 0);
    CHECK(castSca<int64_t, bool>(true) == 1);
    CHECK(castSca<double, bool>(false) == 0.0);
    CHECK(castSca<double, bool>(true) == 1.0);
    
    CHECK(castSca<bool, int64_t>(123) == true);
    CHECK(castSca<bool, int64_t>(-123) == true);
    CHECK(castSca<bool, int64_t>(0) == false);
    CHECK(castSca<bool, double>(123.4) == true);
    CHECK(castSca<bool, double>(-123.4) == true);
    CHECK(castSca<bool, double>(0.0) == false);
}