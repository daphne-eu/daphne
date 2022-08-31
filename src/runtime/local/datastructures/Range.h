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

#pragma once

#include <memory>

// Unused for now. This can be used to track sub allocations of matrices
struct Range {
    size_t r_start;
    size_t c_start;
    size_t r_len;
    size_t c_len;

    explicit Range() : r_start(0), c_start(0), r_len(0), c_len(0) { }
    explicit Range(size_t r1, size_t c1, size_t r2, size_t c2) : r_start(r1), c_start(c1), r_len(r2), c_len(c2) { }

    bool operator==(const Range* other) const {
        return((other != nullptr) && (r_start == other->r_start && c_start == other->c_start && r_len == other->r_len &&
                                      c_len == other->c_len));
    }

    bool operator==(const Range other) const {
        return(r_start == other.r_start && c_start == other.c_start && r_len == other.r_len &&
                                      c_len == other.c_len);
    }

    [[nodiscard]] std::unique_ptr<Range> clone() const { return std::make_unique<Range>(*this); }
};
