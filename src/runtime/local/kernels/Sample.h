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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SAMPLEOP_H
#define SRC_RUNTIME_LOCAL_KERNELS_SAMPLEOP_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <unordered_set>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct Sample {
    static void apply(DTRes *& res, VTArg range, size_t size, bool withReplacement, int64_t seed, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void sample(DTRes *& res, VTArg range, size_t size, bool withReplacement, int64_t seed, DCTX(ctx)) {
    Sample<DTRes, VTArg>::apply(res, range, size, withReplacement, seed, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Sample<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, VT range, int64_t size, bool withReplacement, int64_t seed, DCTX(ctx)) {
        if (size <= 0)
            throw std::runtime_error("size (rows) must be > 0");
        if (range <= 0)
            throw std::runtime_error("range must be > 0");
        if (!withReplacement && !std::is_floating_point<VT>::value &&
            range < size) {
            throw std::runtime_error("if no duplicates are allowed, "
                                     "then must be range >= size");
        }

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(size, 1, false);

        if (seed == -1) {
            std::random_device rd;
            std::uniform_int_distribution<int64_t> seedRnd;
            seed = seedRnd(rd);
        }

        std::mt19937 genVal(seed);

        if (!std::is_floating_point<VT>::value && !std::is_integral<VT>::value)
            throw std::runtime_error(
                "the value type must be either floating point or integral");

        // TODO For std::uniform_real_distribution, the upper bound is not
        // included in the interval of possible values, so when VT is a
        // floating-point type, std::nextafter() is not required. However, we
        // don't lose much by that, so it is fine for now.
        typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>>::type distrVal(0, std::nextafter(range, 0));
        if (withReplacement) {            

            VT *valuesRes = res->getValues();
            for (int64_t c = 0; c < size; c++)
            {
                valuesRes[c] = distrVal(genVal);
            }
        }
        else {
            // If range is `double` we can simply store each number we 
            // generate and check if it already exists each time (doubles
            // are rarely duplicate).
            if (std::is_floating_point<VT>::value){
                
                std::unordered_set<VT> contained;
                VT *valuesRes = res->getValues();
                for (int64_t c = 0; c < size; c++)
                {
                    VT generatedValue = distrVal(genVal);
                    while (contained.find(generatedValue) != contained.end()){
                        generatedValue = distrVal(genVal);                        
                    }                    
                    valuesRes[c] = generatedValue;
                    contained.insert(generatedValue);
                }   
            }
            // Else if range is `int` the above method does not work efficiently.
            // Ex. size = range, finding the correct number is increasingly
            // harder as we fill the array. We must implement an efficient algorithm
            // to create non-duplicate numbers (see Knuth's algorithm).
            else {                
                VT *valuesRes = res->getValues();
                int64_t iRange, iSize;
                iSize = 0;

                for (iRange = 0; iRange < range && iSize < size; iRange++) {
                    size_t rRange = range - iRange;
                    size_t rSize = size - iSize;
                    if (fmod(distrVal(genVal), rRange) < rSize)
                        valuesRes[iSize++] = iRange;
                }
                std::shuffle(valuesRes, valuesRes + size, genVal);
            }
        }
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_SAMPLEOP_H
