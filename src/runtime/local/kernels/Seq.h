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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdlib.h>
#include <cmath>
#include <iomanip>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Seq{
    static void apply(DT *& res, typename DT::VT start,typename DT::VT end, typename DT::VT inc, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void seq(DT *& res, typename DT::VT start, typename DT::VT end, typename DT::VT inc, DCTX(ctx)) {
    Seq<DT>::apply(res, start, end, inc, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VT>
struct Seq<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, VT start, VT end, VT inc, DCTX(ctx)) {
        if (std::isnan(inc))
            throw std::runtime_error("inc cannot be NaN");
        if (std::isnan(start))
            throw std::runtime_error("start cannot be NaN");
        if (std::isnan(end))
            throw std::runtime_error("end cannot be NaN");
        if (inc == 0)
            throw std::runtime_error("inc should not be zero");

        if( (start<end && inc<0) || (start>end && inc>0)){
            // Return matrix with zero rows.
            res = DataObjectFactory::create<DenseMatrix<VT>>(0, 1, false);
            return;
        }
        
        VT initialDistanceToEnd= abs(end-start);
        const size_t expectedNumRows= ceil((initialDistanceToEnd/abs(inc)))+1; // number of steps = expectedNumRows and numRows might = expectedNumRows -1 ot expectedNumRows
        const size_t numCols=1;
        // should the kernel do such a check or reallocate res matrix directly?
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(expectedNumRows, numCols, false);
        else if (res->getNumRows() != expectedNumRows)
            throw std::runtime_error(
                "input matrix is not null and may not fit the sequence");

        VT * allValues= res->getValues();

        VT accumulatorValue= start;

        for(size_t i =0; i<expectedNumRows; i++){
            allValues[i]= accumulatorValue;
            accumulatorValue+=inc;
        }

        VT lastValue=allValues[expectedNumRows-1];

        VT eps = 1.0e-13;

        // on my machine the difference is (1.7e-15) greater  than epsilon std::numeric_limits<VT>::epsilon() 
        if ( (end < start) && end-lastValue>eps ) { // reversed sequence
            res->shrinkNumRows(expectedNumRows-1);
        }
        else if ( (end > start) && lastValue-end> eps ){ // normal sequence
            res->shrinkNumRows(expectedNumRows-1);
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SEQ_H
