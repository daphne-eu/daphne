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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_HASSPECIALVALUE_H
#define SRC_RUNTIME_LOCAL_KERNELS_HASSPECIALVALUE_H

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <string>
#include <cmath>
#include <type_traits>
template <class DTArg> struct HasSpecialValue {
    static bool apply(const DTArg *arg) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks the matrix for the special values infinity and NaN.
 *
 * Checks each element of the matrix for:
 *
 *  - std::numeric_limits<double_t>::signaling_NaN();
 *  - std::numeric_limits<double_t>::quiet_NaN();
 *  - std::numeric_limits<double_t>::infinity();
 *
 * @param arg Pointer to the to check.
 * @return Returns true on first find of inifinity or NaN.
 */
template <class DTArg> bool hasSpecialValue(const DTArg *arg) {
    return HasSpecialValue<DTArg>::apply(arg);
}

template <typename VT> struct HasSpecialValue<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> *arg) {

        if (!std::is_floating_point<VT>::value) return false;

        auto values = arg->getValues();
        size_t numValues = arg->getNumRows() * arg->getNumCols();

        for (size_t idx = 0; idx < numValues; idx++) {
            const VT * value = values + idx;
            if (!std::isnan(*value) || !std::isinf(*value)) {
                return true;
            }
        }

        return false;
    }
};

template <typename VT> struct HasSpecialValue<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> *arg) {

        if (!std::is_floating_point<VT>::value) return false;

        auto values = arg->getValues();
        size_t numValues = arg->getNumRows() * arg->getNumCols();

        for (size_t idx = 0; idx < numValues; idx++) {
            const VT * value = values + idx;
            if (!std::isnan(*value) || !std::isinf(*value)) {
                return true;
            }
        }

        return false;
    }
};

#endif
