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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CREATEFRAME_H
#define SRC_RUNTIME_LOCAL_KERNELS_CREATEFRAME_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>

#include <vector>

#include <cstddef>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void createFrame(Frame *&res, Structure **colMats, size_t numColMats, const char **labels, size_t numLabels,
                 DCTX(ctx)) {
    std::vector<Structure *> colMatsVec;
    for (size_t c = 0; c < numColMats; c++)
        colMatsVec.push_back(colMats[c]);

    std::string *labelsStr = numLabels ? new std::string[numLabels] : nullptr;
    for (size_t c = 0; c < numLabels; c++)
        labelsStr[c] = labels[c];

    res = DataObjectFactory::create<Frame>(colMatsVec, labelsStr);

    if (numLabels)
        delete[] labelsStr;
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_CREATEFRAME_H