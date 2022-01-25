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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PRINTSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_PRINTSCA_H

#include <runtime/local/context/DaphneContext.h>

#include <iostream>

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Prints a scalar value to standard output.
 *
 * @param arg The value to print.
 */
template<typename VT>
void printSca(VT arg, bool newline, bool err, DCTX(ctx)) {
    std::ostream & os = err ? std::cerr : std::cout;
    os << arg;
    if(newline)
        os << std::endl;
}

//For printing int8_t/uint8_t as numbers as opposed to characters
template<>
void printSca(int8_t arg, bool newline, bool err, DCTX(ctx)) {
    std::ostream & os = err ? std::cerr : std::cout;
    os << static_cast<int32_t>(arg);
    if(newline)
        os << std::endl;
}
template<>
void printSca(uint8_t arg, bool newline, bool err, DCTX(ctx)) {
    std::ostream & os = err ? std::cerr : std::cout;
    os << static_cast<int32_t>(arg);
    if(newline)
        os << std::endl;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_PRINTSCA_H
