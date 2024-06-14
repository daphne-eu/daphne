/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <runtime/local/context/DaphneContext.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg>
struct TypeOf {
    static void apply(char *& res, const DTArg * arg, DCTX(ctx)) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg>
void typeof(char *& res, const DTArg * arg, DCTX(ctx)) {
    TypeOf<DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// Template specializations for different data/value types
// ****************************************************************************

template<>
struct TypeOf<size_t> {
    static void apply(char *& res, const size_t *arg, DCTX(ctx)) {
        std::memcpy(res, "size_t", 7);
    }
};

template<>
struct TypeOf<double> {
    static void apply(char *& res, const double *arg, DCTX(ctx)) {
        std::memcpy(res, "double", 7);
    }
};

template<>
struct TypeOf<float> {
    static void apply(char *& res, const float *arg, DCTX(ctx)) {
        std::memcpy(res, "float", 6);
    }
};

template<>
struct TypeOf<int64_t> {
    static void apply(char *& res, const int64_t *arg, DCTX(ctx)) {
        std::memcpy(res, "int64_t", 8);
    }
};

template<>
struct TypeOf<int32_t> {
    static void apply(char *& res, const int32_t *arg, DCTX(ctx)) {
        std::memcpy(res, "int32_t", 8);
    }
};

template<>
struct TypeOf<int8_t> {
    static void apply(char *& res, const int8_t *arg, DCTX(ctx)) {
        std::memcpy(res, "int8_t", 6);
    }
};

template<>
struct TypeOf<uint64_t> {
    static void apply(char *& res, const uint64_t *arg, DCTX(ctx)) {
        std::memcpy(res, "uint64_t", 9);
    }
};

template<>
struct TypeOf<uint32_t> {
    static void apply(char *& res, const uint32_t *arg, DCTX(ctx)) {
        std::memcpy(res, "uint32_t", 9);
    }
};

template<>
struct TypeOf<uint8_t> {
    static void apply(char *& res, const uint8_t *arg, DCTX(ctx)) {
        std::memcpy(res, "uint8_t", 7);
    }
};

template<>
struct TypeOf<bool> {
    static void apply(char *& res, const bool *arg, DCTX(ctx)) {
        std::memcpy(res, "bool", 5);
    }
};