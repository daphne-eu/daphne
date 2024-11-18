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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CASTSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_CASTSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <cstring>

#include <string>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

/**
 * @brief Casts the given scalar to another type.
 *
 * @param arg The value to cast.
 * @return The casted value.
 */
template <typename VTRes, typename VTArg> struct CastSca {
    static VTRes apply(VTArg arg, DCTX(ctx)) {
        // Default implementation.
        return static_cast<VTRes>(arg);
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTRes, typename VTArg> VTRes castSca(VTArg arg, DCTX(ctx)) {
    return CastSca<VTRes, VTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// string <- any type
// ----------------------------------------------------------------------------

template <typename VTArg> struct CastSca<const char *, VTArg> {
    static const char *apply(VTArg arg, DCTX(ctx)) {
        std::string str = std::to_string(arg).c_str();
        const size_t len = str.length();
        char *res = new char[len + 1]();
        strncpy(res, str.c_str(), len);
        res[len] = 0;
        return res;
    }
};

template <typename VTArg> struct CastSca<std::string, VTArg> {
    static std::string apply(VTArg arg, DCTX(ctx)) { return std::to_string(arg).c_str(); }
};

// ----------------------------------------------------------------------------
// any type <- string
// ----------------------------------------------------------------------------

template <typename VTRes> struct CastSca<VTRes, std::string> {
    static VTRes apply(std::string arg, DCTX(ctx)) {
        if constexpr (std::is_integral<VTRes>::value) {
            if constexpr (std::is_unsigned<VTRes>::value)
                return static_cast<VTRes>(std::stoull(arg));
            else
                return static_cast<VTRes>(std::stoll(arg));
        } else if constexpr (std::is_same<VTRes, double>::value)
            return static_cast<VTRes>(std::stold(arg));

        else if constexpr (std::is_same<VTRes, float>::value)
            return static_cast<VTRes>(std::stof(arg));
        else {
            // Trigger a compiler warning using deprecated attribute.
            return throwUnsupportedType(arg);
        }
    }

    [[deprecated("CastSca: Warning! Unsupported result type in casting string values.")]]
    static VTRes throwUnsupportedType(std::string arg) {
        throw std::runtime_error("CastSca: Unsupported result type in casting string values");
    }
};

template <typename VTRes> struct CastSca<VTRes, FixedStr16> {
    static VTRes apply(FixedStr16 arg, DCTX(ctx)) {
        if constexpr (std::is_integral<VTRes>::value) {
            if constexpr (std::is_unsigned<VTRes>::value)
                return static_cast<VTRes>(std::stoull(arg.buffer));
            else
                return static_cast<VTRes>(std::stoll(arg.buffer));
        } else if constexpr (std::is_same<VTRes, double>::value)
            return static_cast<VTRes>(std::stold(arg.buffer));
        else if constexpr (std::is_same<VTRes, float>::value)
            return static_cast<VTRes>(std::stof(arg.buffer));
        else {
            // Trigger a compiler warning using deprecated attribute.
            return throwUnsupportedType(arg);
        }
    }

    [[deprecated("CastSca: Warning! Unsupported result type in casting string values.")]]
    static VTRes throwUnsupportedType(std::string arg) {
        throw std::runtime_error("CastSca: Unsupported result type in casting string values");
    }
};

// ----------------------------------------------------------------------------
// string <- string
// ----------------------------------------------------------------------------
template <> struct CastSca<std::string, std::string> {
    static std::string apply(const std::string &arg, DaphneContext *ctx) { return arg; }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_CASTSCA_H
