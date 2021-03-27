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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPEUTILS_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPEUTILS_H

#include <runtime/local/datastructures/ValueTypeCode.h>

#include <cinttypes>
#include <cstddef>

struct ValueTypeUtils {

    static size_t sizeOf(ValueTypeCode type) {
        switch(type) {
            case ValueTypeCode::SI8:  return sizeof(int8_t);
            case ValueTypeCode::SI32: return sizeof(int32_t);
            case ValueTypeCode::SI64: return sizeof(int64_t);
            case ValueTypeCode::UI8:  return sizeof(uint8_t);
            case ValueTypeCode::UI32: return sizeof(uint32_t);
            case ValueTypeCode::UI64: return sizeof(uint64_t);
            case ValueTypeCode::F32: return sizeof(float);
            case ValueTypeCode::F64: return sizeof(double);
            default: throw std::runtime_error("unknown value type code");
        }
    }

    template<typename ValueType>
    static const ValueTypeCode codeFor;

};

template<>
const ValueTypeCode ValueTypeUtils::codeFor<int8_t>   = ValueTypeCode::SI8;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<int32_t>  = ValueTypeCode::SI32;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<int64_t>  = ValueTypeCode::SI64;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<uint8_t>  = ValueTypeCode::UI8;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<uint32_t> = ValueTypeCode::UI32;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<uint64_t> = ValueTypeCode::UI64;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<float>  = ValueTypeCode::F32;

template<>
const ValueTypeCode ValueTypeUtils::codeFor<double> = ValueTypeCode::F64;

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPEUTILS_H

