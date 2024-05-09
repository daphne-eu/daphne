/*
 * Copyright 2024 The DAPHNE Consortium
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

// TODO Should be <daphne/runtime/local/datastructures/DenseMatrix.h> for more clarity.
#include <runtime/local/datastructures/DenseMatrix.h>

#include <iostream>

#include <cstdlib>

class DaphneContext;

extern "C" {

    void mySumAll(float* res, const DenseMatrix<float>* arg, DaphneContext* ctx) {
        std::cout << "hello from mySumAll" << std::endl;
        const float* valuesArg = arg->getValues();
        *res = 0;
        for(size_t r = 0; r < arg->getNumRows(); r++) {
            for(size_t c = 0; c < arg->getNumCols(); c++)
                *res += valuesArg[c];
            valuesArg += arg->getRowSkip();
        }
    }

}
