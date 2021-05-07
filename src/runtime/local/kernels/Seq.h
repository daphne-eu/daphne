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

#include <type_traits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VT, typename VT, typename VT>
struct Seq{
    static void apply(DTRes *& res, VT start, VT end, VT incp) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VT, typename VT, typename VT>
void seq(DTRes *& res, VT start, VT end, VT inc) {
    Seq<DTRes, VT, VT, VT>::apply(res, start, end, inc);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VT>
struct Seqx<DenseMatrix<VT>, VT, VT,VT> {
    static void apply(DenseMatrix<VT> *& res, VT start, VT end, VT inc) {
        assert(start<end 0 && "start must be less than end");
        assert(inc>0 && "this inc must be greater than zero");
	assert(res==nullptr && "result matrix should point to null")
	
	size_t numRows= (int64_t)(((end-start)/inc) +1);
	size_t numCols=1;
	size_t rowSkip	
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
	VT * allValues= res->getValues();
	const size_t rowSkip= res->getRowSkip();
	VT accumulatorValue= start;
	for(size_t i =0; i<numRows ;i++){
		allvalues[i*rowSkip]= accumulatorvalue;
		accumulatorValue+=inc;
	}
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SEQ_H
