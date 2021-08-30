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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PRINTOBJ_H
#define SRC_RUNTIME_LOCAL_KERNELS_PRINTOBJ_H

#include <runtime/local/context/DaphneContext.h>

#include <iostream>
#include <iomanip>

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Prints a data object to standard output.
 * 
 * Template paramter `DT` should be a sub-class of `Structure`, e.g.
 * `DenseMatrix`, `CSRMatrix`, or `Frame`.
 * 
 * @param arg The data object to print.
 */
template<class DT>
void printObj(const DT * arg, bool newline, bool err, DCTX(ctx)) {
    arg->print(err ? std::cerr : std::cout);
}

template<>
void printObj(const char * arg, bool newline, bool err, DCTX(ctx)) {
    std::ostream & os = err ? std::cerr : std::cout;
    os << std::setprecision(10) << std::fixed << arg;
    if(newline)
        os << std::endl;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_PRINTOBJ_H