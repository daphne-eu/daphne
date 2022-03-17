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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INCREFKERNELRES_H
#define SRC_RUNTIME_LOCAL_KERNELS_INCREFKERNELRES_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Structure.h>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void incRefKernelRes(
        const Structure ** kernelRess, size_t numKernelRess,
        const Structure ** kernelArgs, size_t numKernelArgs,
        DCTX(ctx)
) {
    for(size_t i = 0; i < numKernelRess; i++)
        for(size_t j = 0; j < numKernelArgs; j++)
            if(kernelRess[i] == kernelArgs[j]) {
                kernelRess[i]->increaseRefCounter();
                break;
            }
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_INCREFKERNELRES_H