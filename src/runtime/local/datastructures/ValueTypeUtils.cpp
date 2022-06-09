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

#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/datastructures/ValueTypeCode.h>

#include <stdexcept>
#include <string>

#include <cstddef>
#include <cstdint>

size_t ValueTypeUtils::sizeOf(ValueTypeCode type) {
    switch(type) {
        case ValueTypeCode::SI8:  return sizeof(int8_t);
        case ValueTypeCode::SI32: return sizeof(int32_t);
        case ValueTypeCode::SI64: return sizeof(int64_t);
        case ValueTypeCode::UI8:  return sizeof(uint8_t);
        case ValueTypeCode::UI32: return sizeof(uint32_t);
        case ValueTypeCode::UI64: return sizeof(uint64_t);
        case ValueTypeCode::F32: return sizeof(float);
        case ValueTypeCode::F64: return sizeof(double);
        default: throw std::runtime_error("ValueTypeUtils::sizeOf: unknown value type code");
    }
}

void ValueTypeUtils::printValue(std::ostream & os, ValueTypeCode type, const void * array, size_t pos) {
    switch(type) {
        //Conversion int8->int32 for formating as number as opposed to character
        case ValueTypeCode::SI8:  os << static_cast<int32_t>(reinterpret_cast<const int8_t  *>(array)[pos]); break;
        case ValueTypeCode::SI32: os << reinterpret_cast<const int32_t *>(array)[pos]; break;
        case ValueTypeCode::SI64: os << reinterpret_cast<const int64_t *>(array)[pos]; break;
        //Conversion uint8->uint32 for formating as number as opposed to character
        case ValueTypeCode::UI8:  os << static_cast<uint32_t>(reinterpret_cast<const uint8_t  *>(array)[pos]); break;
        case ValueTypeCode::UI32: os << reinterpret_cast<const uint32_t *>(array)[pos]; break;
        case ValueTypeCode::UI64: os << reinterpret_cast<const uint64_t *>(array)[pos]; break;
        case ValueTypeCode::F32: os << reinterpret_cast<const float  *>(array)[pos]; break;
        case ValueTypeCode::F64: os << reinterpret_cast<const double *>(array)[pos]; break;
        default: throw std::runtime_error("ValueTypeUtils::printValue: unknown value type code");
    }
}

template<> const ValueTypeCode ValueTypeUtils::codeFor<int8_t>   = ValueTypeCode::SI8;
template<> const ValueTypeCode ValueTypeUtils::codeFor<int32_t>  = ValueTypeCode::SI32;
template<> const ValueTypeCode ValueTypeUtils::codeFor<int64_t>  = ValueTypeCode::SI64;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint8_t>  = ValueTypeCode::UI8;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint32_t> = ValueTypeCode::UI32;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint64_t> = ValueTypeCode::UI64;
template<> const ValueTypeCode ValueTypeUtils::codeFor<float>  = ValueTypeCode::F32;
template<> const ValueTypeCode ValueTypeUtils::codeFor<double> = ValueTypeCode::F64;

template<> const std::string ValueTypeUtils::cppNameFor<int8_t>   = "int8_t";
template<> const std::string ValueTypeUtils::cppNameFor<int32_t>  = "int32_t";
template<> const std::string ValueTypeUtils::cppNameFor<int64_t>  = "int64_t";
template<> const std::string ValueTypeUtils::cppNameFor<uint8_t>  = "uint8_t";
template<> const std::string ValueTypeUtils::cppNameFor<uint32_t> = "uint32_t";
template<> const std::string ValueTypeUtils::cppNameFor<uint64_t> = "uint64_t";
template<> const std::string ValueTypeUtils::cppNameFor<float>  = "float";
template<> const std::string ValueTypeUtils::cppNameFor<double> = "double";
template<> const std::string ValueTypeUtils::cppNameFor<bool> = "bool";

template<> const std::string ValueTypeUtils::irNameFor<int8_t>   = "si8";
template<> const std::string ValueTypeUtils::irNameFor<int32_t>  = "si32";
template<> const std::string ValueTypeUtils::irNameFor<int64_t>  = "si64";
template<> const std::string ValueTypeUtils::irNameFor<uint8_t>  = "ui8";
template<> const std::string ValueTypeUtils::irNameFor<uint32_t> = "ui32";
template<> const std::string ValueTypeUtils::irNameFor<uint64_t> = "ui64";
template<> const std::string ValueTypeUtils::irNameFor<float>  = "f32";
template<> const std::string ValueTypeUtils::irNameFor<double> = "f64";
    
const std::string ValueTypeUtils::cppNameForCode(ValueTypeCode type) {
    switch(type) {
        case ValueTypeCode::SI8:  return cppNameFor<int8_t>;
        case ValueTypeCode::SI32: return cppNameFor<int32_t>;
        case ValueTypeCode::SI64: return cppNameFor<int64_t>;
        case ValueTypeCode::UI8:  return cppNameFor<uint8_t>;
        case ValueTypeCode::UI32: return cppNameFor<uint32_t>;
        case ValueTypeCode::UI64: return cppNameFor<uint64_t>;
        case ValueTypeCode::F32: return cppNameFor<float>;
        case ValueTypeCode::F64: return cppNameFor<double>;
        default: throw std::runtime_error("ValueTypeUtils::cppNameForCode: unknown value type code");
    }
}

const std::string ValueTypeUtils::irNameForCode(ValueTypeCode type) {
    switch(type) {
        case ValueTypeCode::SI8:  return irNameFor<int8_t>;
        case ValueTypeCode::SI32: return irNameFor<int32_t>;
        case ValueTypeCode::SI64: return irNameFor<int64_t>;
        case ValueTypeCode::UI8:  return irNameFor<uint8_t>;
        case ValueTypeCode::UI32: return irNameFor<uint32_t>;
        case ValueTypeCode::UI64: return irNameFor<uint64_t>;
        case ValueTypeCode::F32: return irNameFor<float>;
        case ValueTypeCode::F64: return irNameFor<double>;
        default: throw std::runtime_error("ValueTypeUtils::irNameForCode: unknown value type code");
    }
}
