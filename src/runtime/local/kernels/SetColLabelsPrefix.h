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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELSPREFIX_H
#define SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELSPREFIX_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/LabelUtils.h>

#include <string>

#include <cstddef>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void setColLabelsPrefix(Frame *& res, const Frame * arg, const char * prefix, DCTX(ctx)) {
    const size_t numCols = arg->getNumCols();
    const std::string * oldLabels = arg->getLabels();
    std::string * newLabels = new std::string[numCols];
    
    for(size_t i = 0; i < numCols; i++)
        newLabels[i] = LabelUtils::setPrefix(prefix, oldLabels[i]);
    
    // Create a view on the input frame (zero-copy) and modify the column
    // labels of the view.
    auto colIdxs = new size_t[numCols];
    for(size_t c = 0; c < numCols; c++)
        colIdxs[c] = c;
    res = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), numCols, colIdxs);
    delete[] colIdxs;
    res->setLabels(newLabels);
    
    delete[] newLabels;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELSPREFIX_H