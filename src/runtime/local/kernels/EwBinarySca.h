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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWBINARYSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWBINARYSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/BinaryOpCode.h>

#include <algorithm>
#include <stdexcept>

#include <cmath>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<BinaryOpCode opCode, class VTRes, class VTLhs, class VTRhs>
// Note that, deviating from the kernel function ewBinarySca below, the opCode
// is a template parameter here, because we want to enable re-use for efficient
// elementwise operations on matrices, where we want to be able to avoid the
// overhead of interpreting the opCode for each pair of values at run-time.
struct EwBinarySca {
    static VTRes apply(VTLhs lhs, VTRhs rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Function pointers for binary functions
// ****************************************************************************

/**
 * @brief A function pointer to a binary function on scalars.
 */
template<typename VTRes, typename VTLhs, typename VTRhs>
using EwBinaryScaFuncPtr = VTRes (*)(VTLhs, VTRhs, DCTX());

/**
 * @brief Returns the binary function on scalars for the specified binary
 * operation.
 * 
 * @param opCode
 * @return 
 */
template<typename VTRes, typename VTLhs, typename VTRhs>
EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> getEwBinaryScaFuncPtr(BinaryOpCode opCode) {
    switch (opCode) {
#define MAKE_CASE(opCode) case opCode: return &EwBinarySca<opCode, VTRes, VTLhs, VTRhs>::apply;
        // Arithmetic.
        MAKE_CASE(BinaryOpCode::ADD)
        MAKE_CASE(BinaryOpCode::SUB)
        MAKE_CASE(BinaryOpCode::MUL)
        MAKE_CASE(BinaryOpCode::DIV)
        MAKE_CASE(BinaryOpCode::POW)
        MAKE_CASE(BinaryOpCode::MOD)
        MAKE_CASE(BinaryOpCode::LOG)
        // Comparisons.
        MAKE_CASE(BinaryOpCode::EQ)
        MAKE_CASE(BinaryOpCode::NEQ)
        MAKE_CASE(BinaryOpCode::LT)
        MAKE_CASE(BinaryOpCode::LE)
        MAKE_CASE(BinaryOpCode::GT)
        MAKE_CASE(BinaryOpCode::GE)
        // Min/max.
        MAKE_CASE(BinaryOpCode::MIN)
        MAKE_CASE(BinaryOpCode::MAX)
        // Logical.
        MAKE_CASE(BinaryOpCode::AND)
        MAKE_CASE(BinaryOpCode::OR)
#undef MAKE_CASE
        default:
            throw std::runtime_error("unknown BinaryOpCode");
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Performs a binary operation on two scalars.
 * 
 * @param opCode The binary operation to perform.
 * @param lhs The left-hand-side operand.
 * @param rhs The right-hand-side operand.
 * @return The result of the binary operation.
 */
template<typename TRes, typename TLhs, typename TRhs>
TRes ewBinarySca(BinaryOpCode opCode, TLhs lhs, TRhs rhs, DCTX(ctx)) {
    return getEwBinaryScaFuncPtr<TRes, TLhs, TRhs>(opCode)(lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different op codes
// ****************************************************************************

// Handle multiply extra
template<typename TLhs, typename TRhs>
struct EwBinarySca<BinaryOpCode::MUL, bool, TLhs, TRhs> {
    inline static bool apply(TLhs lhs, TRhs rhs, DCTX(ctx)) {
        uint32_t result = lhs * rhs;
        return static_cast<bool>(result);
    }
};

#define MAKE_EW_BINARY_SCA(opCode, expr) \
    template<typename TRes, typename TLhs, typename TRhs> \
    struct EwBinarySca<opCode, TRes, TLhs, TRhs> { \
        inline static TRes apply(TLhs lhs, TRhs rhs, DCTX(ctx)) { \
            return expr; \
        } \
    };

// One such line for each binary function to support.
// Arithmetic.
MAKE_EW_BINARY_SCA(BinaryOpCode::ADD, lhs + rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::SUB, lhs - rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::MUL, lhs * rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::DIV, lhs / rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::POW, pow(lhs, rhs))
MAKE_EW_BINARY_SCA(BinaryOpCode::MOD, std::fmod(lhs, rhs))
MAKE_EW_BINARY_SCA(BinaryOpCode::LOG, std::log(lhs)/std::log(rhs))
// Comparisons.
MAKE_EW_BINARY_SCA(BinaryOpCode::EQ , lhs == rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::NEQ, lhs != rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::LT , lhs <  rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::LE , lhs <= rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::GT , lhs >  rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::GE , lhs >= rhs)
// Min/max.
MAKE_EW_BINARY_SCA(BinaryOpCode::MIN, std::min(lhs, rhs))
MAKE_EW_BINARY_SCA(BinaryOpCode::MAX, std::max(lhs, rhs))
// Logical.
MAKE_EW_BINARY_SCA(BinaryOpCode::AND, lhs && rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::OR , lhs || rhs)

#undef MAKE_EW_BINARY_SCA

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWBINARYSCA_H
