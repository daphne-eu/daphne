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

#include <runtime/local/kernels/BinaryOpCode.h>

#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<BinaryOpCode opCode, class VTRes, class VTLhs, class VTRhs>
struct EwBinarySca {
    static VTRes apply(VTLhs lhs, VTRhs rhs) = delete;
};

// ****************************************************************************
// Function pointers for binary functions
// ****************************************************************************

/**
 * @brief A function pointer to a binary function on scalars.
 */
template<typename VTRes, typename VTLhs, typename VTRhs>
using EwBinaryScaFuncPtr = VTRes (*)(VTLhs, VTRhs);

/**
 * @brief Returns the binary function on scalars for the specified binary
 * operation.
 * 
 * @param opCode
 * @return 
 */
template<typename VTRes, typename VTLhs, typename VTRhs>
EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> getEwBinaryScaFuncPtr(BinaryOpCode opCode) {
    switch(opCode) {
        #define MAKE_CASE(opCode) case opCode: return &EwBinarySca<opCode, VTRes, VTLhs, VTRhs>::apply;
        MAKE_CASE(BinaryOpCode::ADD)
        MAKE_CASE(BinaryOpCode::MUL)
        #undef MAKE_CASE
        default:
            throw std::runtime_error("unknown BinaryOpCode");
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Performs a binary operation on two scalars, whereby the operation is
 * known at compile-time.
 * 
 * Thus, `opCode` is a template parameter here.
 * 
 * @param lhs The left-hand-side operand.
 * @param rhs The right-hand-side operand.
 * @return The result of the binary operation.
 */
template<BinaryOpCode opCode, class TRes, class TLhs, class TRhs>
TRes ewBinaryScaCT(TLhs lhs, TRhs rhs) {
    return EwBinarySca<opCode, TRes, TLhs, TRhs>::apply(lhs, rhs);
}

/**
 * @brief Performs a binary operation on two scalars, whereby the operation is
 * only known at run-time.
 * 
 * Thus, `opCode` is a run-time parameter here.
 * 
 * @param opCode The binary operation to perform.
 * @param lhs The left-hand-side operand.
 * @param rhs The right-hand-side operand.
 * @return The result of the binary operation.
 */
template<typename TRes, typename TLhs, typename TRhs>
TRes ewBinaryScaRT(BinaryOpCode opCode, TLhs lhs, TRhs rhs) {
    return getEwBinaryScaFuncPtr<TRes, TLhs, TRhs>(opCode)(lhs, rhs);
}

// ****************************************************************************
// (Partial) template specializations for different op codes
// ****************************************************************************

#define MAKE_EW_BINARY_SCA(opCode, expr) \
    template<typename TRes, typename TLhs, typename TRhs> \
    struct EwBinarySca<opCode, TRes, TLhs, TRhs> { \
        inline static TRes apply(TLhs lhs, TRhs rhs) { \
            return expr; \
        } \
    };

// One such line for each binary function to support.
MAKE_EW_BINARY_SCA(BinaryOpCode::ADD, lhs + rhs)
MAKE_EW_BINARY_SCA(BinaryOpCode::MUL, lhs * rhs)

#undef MAKE_EW_BINARY_SCA

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWBINARYSCA_H