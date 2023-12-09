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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/UnaryOpCode.h>

#include <limits>
#include <stdexcept>
#include <sstream>

#include <cmath>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<UnaryOpCode opCode, class VTRes, class VTArg>
// Note that, deviating from the kernel function ewUnarySca below, the opCode
// is a template parameter here, because we want to enable re-use for efficient
// elementwise operations on matrices, where we want to be able to avoid the
// overhead of interpreting the opCode for each value at run-time.
struct EwUnarySca {
    static VTRes apply(VTArg arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Function pointers for unary functions
// ****************************************************************************

/**
 * @brief A function pointer to a unary function on scalars.
 */
template<typename VTRes, typename VTArg>
using EwUnaryScaFuncPtr = VTRes (*)(VTArg, DCTX());

/**
 * @brief Returns the unary function on scalars for the specified unary
 * operation.
 * 
 * @param opCode
 * @return 
 */
template<typename VTRes, typename VTArg>
EwUnaryScaFuncPtr<VTRes, VTArg> getEwUnaryScaFuncPtr(UnaryOpCode opCode) {
    switch(opCode) {
        #define MAKE_CASE(opCode) case opCode: return &EwUnarySca<opCode, VTRes, VTArg>::apply;
        // Arithmetic/general math.
        MAKE_CASE(UnaryOpCode::ABS)
        MAKE_CASE(UnaryOpCode::SIGN)
        MAKE_CASE(UnaryOpCode::SQRT)
        MAKE_CASE(UnaryOpCode::EXP)
        MAKE_CASE(UnaryOpCode::LN)
        // Trigonometric/Hyperbolic functions
        MAKE_CASE(UnaryOpCode::SIN)
        MAKE_CASE(UnaryOpCode::COS)
        MAKE_CASE(UnaryOpCode::TAN)
        MAKE_CASE(UnaryOpCode::ASIN)
        MAKE_CASE(UnaryOpCode::ACOS)
        MAKE_CASE(UnaryOpCode::ATAN)
        MAKE_CASE(UnaryOpCode::SINH)
        MAKE_CASE(UnaryOpCode::COSH)
        MAKE_CASE(UnaryOpCode::TANH)
        // Rounding.
        MAKE_CASE(UnaryOpCode::FLOOR)
        MAKE_CASE(UnaryOpCode::CEIL)
        MAKE_CASE(UnaryOpCode::ROUND)
        #undef MAKE_CASE
        default:
            throw std::runtime_error("unknown UnaryOpCode");
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Performs a unary operation on a scalar.
 * 
 * @param opCode The unary operation to perform.
 * @param arg The operand.
 * @return The result of the unary operation.
 */
template<typename TRes, typename TArg>
TRes ewUnarySca(UnaryOpCode opCode, TArg arg, DCTX(ctx)) {
    return getEwUnaryScaFuncPtr<TRes, TArg>(opCode)(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different op codes
// ****************************************************************************

#define MAKE_EW_UNARY_SCA(opCode, expr) \
    template<typename TRes, typename TArg> \
    struct EwUnarySca<opCode, TRes, TArg> { \
        inline static TRes apply(TArg arg, DCTX(ctx)) { \
            return expr; \
        } \
    };

#define MAKE_EW_UNARY_SCA_OPEN_DOMAIN_ERROR(opCode, expr, lowerBound, strFuncDomain) \
    template<typename TRes, typename TArg> \
    struct EwUnarySca<opCode, TRes, TArg> { \
        inline static TRes apply(TArg arg, DCTX(ctx)) { \
            if (lowerBound > arg) { \
                std::ostringstream errMsg; \
                errMsg << "invalid argument '" << arg << "' passed to unary func " << strFuncDomain; \
                throw std::domain_error(errMsg.str()); \
            } \
            return expr; \
        } \
    };

#define MAKE_EW_UNARY_SCA_CLOSED_DOMAIN_ERROR(opCode, expr, lowerBound, upperBound, strFuncDomain) \
    template<typename TRes, typename TArg> \
    struct EwUnarySca<opCode, TRes, TArg> { \
        inline static TRes apply(TArg arg, DCTX(ctx)) { \
            if (lowerBound > arg || arg > upperBound) { \
                std::ostringstream errMsg; \
                errMsg << "invalid argument '" << arg << "' passed to unary func " << strFuncDomain; \
                throw std::domain_error(errMsg.str()); \
            } \
            return expr; \
        } \
    };

// One such line for each unary function to support.
// Arithmetic/general math.
MAKE_EW_UNARY_SCA(UnaryOpCode::ABS, abs(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::SIGN, (arg == 0) ? 0 : ((arg < 0) ? -1 : ((arg > 0) ? 1 : std::numeric_limits<TRes>::quiet_NaN())));
MAKE_EW_UNARY_SCA_OPEN_DOMAIN_ERROR(UnaryOpCode::SQRT, sqrt(arg),
                                    -0.0, "SQRT with domain [-0, inf]")
MAKE_EW_UNARY_SCA(UnaryOpCode::EXP, exp(arg));
MAKE_EW_UNARY_SCA_OPEN_DOMAIN_ERROR(UnaryOpCode::LN, log(arg),
                                    -0.0, "LN with domain [-0, inf]");     // -0 maps to -inf
// Trigonometric/Hyperbolic functions
MAKE_EW_UNARY_SCA(UnaryOpCode::SIN, sin(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::COS, cos(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::TAN, tan(arg));                              // undefined points effectively do not restrict domain
MAKE_EW_UNARY_SCA_CLOSED_DOMAIN_ERROR(UnaryOpCode::ASIN, asin(arg),
                                    -1.0, 1.0, "ASIN with domain [-1, 1]");
MAKE_EW_UNARY_SCA_CLOSED_DOMAIN_ERROR(UnaryOpCode::ACOS, acos(arg),
                                    -1.0, 1.0, "ACOS with domain [-1, 1]");
MAKE_EW_UNARY_SCA(UnaryOpCode::ATAN, atan(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::SINH, sinh(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::COSH, cosh(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::TANH, tanh(arg));
// Rounding.
MAKE_EW_UNARY_SCA(UnaryOpCode::FLOOR, floor(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::CEIL, std::ceil(arg));
MAKE_EW_UNARY_SCA(UnaryOpCode::ROUND, round(arg));

#undef MAKE_EW_UNARY_SCA_CLOSED_DOMAIN_ERROR
#undef MAKE_EW_UNARY_SCA_OPEN_DOMAIN_ERROR
#undef MAKE_EW_UNARY_SCA

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYSCA_H