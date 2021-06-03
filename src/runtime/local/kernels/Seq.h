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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SEQ_H
#define SRC_RUNTIME_LOCAL_KERNELS_SEQ_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdlib.h>
#include <math.h>
#include <cassert>
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VT>
struct Seq{
    static void apply(DTRes *& res, VT start, VT end, VT incp) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VT>
void seq(DTRes *& res, VT start, VT end, VT inc) {
    Seq<DTRes, VT>::apply(res, start, end, inc);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VT>
struct Seq<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, VT start, VT end, VT inc) {
    assert(inc != 0 && "inc should not be zero"); // setp 0 can not make any progress to any given boundary
    assert(start!=end && "start and end should not be equal"); // can not have start = end
    VT distanceToEnd= abs(end-(start+inc)); 
    VT initialDistanceToEnd= abs(end-start);
    assert(distanceToEnd<initialDistanceToEnd  && "repeatedly adding a step to start does not lead to the end"); // to make sure we have a finite sequence
    const size_t numRows= ceil((initialDistanceToEnd/abs(inc))+1); // number of steps = numRows
    const size_t numCols=1;
    if(res == nullptr) // should one do such a check or reallocate directly?
        res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    VT * allValues= res->getValues();
    VT accumulatorValue= start;
    const size_t rowSkip= res->getRowSkip();
    for(size_t i =0; i<numRows ;i++){
      allValues[i*rowSkip]= accumulatorvalue;
      accumulatorValue+=inc;
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SEQ_H
