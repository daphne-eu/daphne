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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/List.h>

#include <iostream>

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
template <class DT> void printObj(const DT *arg, [[maybe_unused]] bool newline, bool err, DCTX(ctx)) {
    arg->print(err ? std::cerr : std::cout);
}

template <> void printObj(const char *arg, bool newline, bool err, DCTX(ctx)) {
    std::ostream &os = err ? std::cerr : std::cout;
    os << arg;
    if (newline)
        os << std::endl;
}
