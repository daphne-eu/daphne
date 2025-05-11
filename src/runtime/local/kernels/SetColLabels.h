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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELS_H
#define SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELS_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>

#include <stdexcept>
#include <string>

#include <cstddef>

// ****************************************************************************
// Convenience function
// ****************************************************************************

inline void setColLabels(Frame *&res, const Frame *arg, const char **labels, size_t numLabels, DCTX(ctx)) {
    const size_t numCols = arg->getNumCols();
    if (numLabels != numCols)
        throw std::runtime_error("the number of given labels (" + std::to_string(numLabels) +
                                 ") does not match the number of columns of the given frame (" +
                                 std::to_string(numCols) + ")");
    std::string *labelsStr = new std::string[numCols];
    for (size_t c = 0; c < numCols; c++)
        labelsStr[c] = labels[c];

    // Create a view on the input frame (zero-copy) and modify the column
    // labels of the view.
    auto colIdxs = new size_t[numCols];
    for (size_t c = 0; c < numCols; c++)
        colIdxs[c] = c;
    res = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), numCols, colIdxs);
    delete[] colIdxs;
    res->setLabels(labelsStr);

    delete[] labelsStr;
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELS_H
