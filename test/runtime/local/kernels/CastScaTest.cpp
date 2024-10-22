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

#include <limits>

#include <cstdint>

TEST_CASE("castSca, no-op casts", TAG_KERNELS) {
    CHECK(castSca<int64_t, int64_t>(123, nullptr) == 123);
    CHECK(castSca<double, double>(123.4, nullptr) == 123.4);
    CHECK(castSca<bool, bool>(false, nullptr) == false);
    CHECK(castSca<bool, bool>(true, nullptr) == true);
}

TEST_CASE("castSca, actual casts", TAG_KERNELS) {
    CHECK(castSca<int64_t, double>(123.4, nullptr) == 123);
    CHECK(castSca<int64_t, double>(-123.4, nullptr) == -123);
    CHECK(castSca<double, int64_t>(123, nullptr) == 123.0);
    CHECK(castSca<double, int64_t>(-123, nullptr) == -123.0);

    CHECK(castSca<int64_t, bool>(false, nullptr) == 0);
    CHECK(castSca<int64_t, bool>(true, nullptr) == 1);
    CHECK(castSca<double, bool>(false, nullptr) == 0.0);
    CHECK(castSca<double, bool>(true, nullptr) == 1.0);

    CHECK(castSca<bool, int64_t>(123, nullptr) == true);
    CHECK(castSca<bool, int64_t>(-123, nullptr) == true);
    CHECK(castSca<bool, int64_t>(0, nullptr) == false);
    CHECK(castSca<bool, double>(123.4, nullptr) == true);
    CHECK(castSca<bool, double>(-123.4, nullptr) == true);
    CHECK(castSca<bool, double>(0.0, nullptr) == false);
}

TEST_CASE("castSca, actual casts strings to numbers", TAG_KERNELS) {

    CHECK(castSca<int64_t, std::string>("123", nullptr) == 123);
    CHECK(castSca<int64_t, std::string>("-123", nullptr) == -123);
    CHECK(castSca<int64_t, std::string>("0", nullptr) == 0);
    CHECK(castSca<double, std::string>("123.4", nullptr) == 123.4);
    CHECK(castSca<double, std::string>("-123.4", nullptr) == -123.4);
    CHECK(castSca<double, std::string>("0.0", nullptr) == 0.0);
    CHECK(castSca<int64_t, std::string>("9223372036854775807", nullptr) == std::numeric_limits<int64_t>::max());
    CHECK(castSca<int64_t, std::string>("-9223372036854775808", nullptr) == std::numeric_limits<int64_t>::min());
    CHECK(castSca<uint64_t, std::string>("18446744073709551615", nullptr) == std::numeric_limits<uint64_t>::max());
    CHECK(castSca<uint64_t, std::string>("0", nullptr) == std::numeric_limits<uint64_t>::min());

    CHECK(castSca<int64_t, FixedStr16>("123", nullptr) == 123);
    CHECK(castSca<int64_t, FixedStr16>("-123", nullptr) == -123);
    CHECK(castSca<int64_t, FixedStr16>("0", nullptr) == 0);
    CHECK(castSca<double, FixedStr16>("123.4", nullptr) == 123.4);
    CHECK(castSca<double, FixedStr16>("-123.4", nullptr) == -123.4);
    CHECK(castSca<double, FixedStr16>("0.0", nullptr) == 0.0);
    CHECK(castSca<int64_t, FixedStr16>("123456789012345", nullptr) == 123456789012345ll);
    CHECK(castSca<int64_t, FixedStr16>("-12345678901234", nullptr) == -12345678901234ll);
    CHECK(castSca<uint64_t, FixedStr16>("123456789012345", nullptr) == 123456789012345ull);
    CHECK(castSca<uint64_t, FixedStr16>("0", nullptr) == std::numeric_limits<uint64_t>::min());
}
