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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>

#include <stdexcept>
#include <string>

#include <cstddef>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void setColLabels(Frame * arg, const char ** labels, size_t numLabels) {
    const size_t numCols = arg->getNumCols();
    if(numLabels != numCols)
        throw std::runtime_error(
                "the number of given labels does not match the number of columns of the given frame"
        );
    std::string * labelsStr = new std::string[numCols];
    for(size_t c = 0; c < numCols; c++)
        labelsStr[c] = labels[c];
    
    arg->setLabels(labelsStr);
    
    delete[] labelsStr;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_SETCOLLABELS_H