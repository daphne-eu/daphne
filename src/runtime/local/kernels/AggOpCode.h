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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGOPCODE_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGOPCODE_H

#include <runtime/local/kernels/BinaryOpCode.h>

#include <limits>
#include <stdexcept>

enum class AggOpCode {
    SUM,
    PROD,
    MIN,
    MAX,
    IDXMIN,
    IDXMAX,
    MEAN,
    STDDEV,
    VAR,
};

struct AggOpCodeUtils {
    static bool isPureBinaryReduction(AggOpCode opCode) {
        switch(opCode) {
            case AggOpCode::SUM:
            case AggOpCode::PROD:
            case AggOpCode::MIN:
            case AggOpCode::MAX:
                return true;
            case AggOpCode::MEAN:
            case AggOpCode::STDDEV:
            case AggOpCode::VAR:
                return false;
            default:
                throw std::runtime_error("unsupported AggOpCode");
        }
    }
    
    static BinaryOpCode getBinaryOpCode(AggOpCode opCode) {
        if (!isPureBinaryReduction(opCode)) {
            throw std::runtime_error(
                "Aggregation kernel expects pure binary reduction.");
        }
        switch(opCode) {
            case AggOpCode::SUM: return BinaryOpCode::ADD;
            case AggOpCode::PROD: return BinaryOpCode::MUL;
            case AggOpCode::MIN: return BinaryOpCode::MIN;
            case AggOpCode::MAX: return BinaryOpCode::MAX;
            default:
                throw std::runtime_error("unsupported AggOpCode");
        }
    }
    
    template<typename VT>
    static VT getNeutral(AggOpCode opCode) {
        if (!isPureBinaryReduction(opCode)) {
            throw std::runtime_error(
                "Aggregation kernel expects pure binary reduction.");
        }
        switch(opCode) {
            case AggOpCode::SUM: return VT(0);
            case AggOpCode::PROD: return VT(1);
            case AggOpCode::MIN: return std::numeric_limits<VT>::has_infinity ?  std::numeric_limits<VT>::infinity() : std::numeric_limits<VT>::max();
            case AggOpCode::MAX: return std::numeric_limits<VT>::has_infinity ? -std::numeric_limits<VT>::infinity() : std::numeric_limits<VT>::min();
            default:
                throw std::runtime_error("unsupported AggOpCode");
        }
    }
    
    static bool isSparseSafe(AggOpCode opCode) {
        switch(opCode) {
            case AggOpCode::SUM:
                return true;
            case AggOpCode::PROD:
            case AggOpCode::MIN:
            case AggOpCode::MAX:
            case AggOpCode::MEAN:
            case AggOpCode::STDDEV:
            case AggOpCode::VAR:
                return false;
            default:
                throw std::runtime_error("unsupported AggOpCode");
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGOPCODE_H
