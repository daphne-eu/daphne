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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <cblas.h>
#include <math.h>

#include <cassert>
#include <cstddef>
#include "syrk_interface.h"

namespace FPGAOPENCL{
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Syrk {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void syrk(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    Syrk<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Syrk<DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

	#define KKK         16
	#define JJJ         8
	#define III         16
	#define JJ          32
	#define II          16
	#define KK          16

    	int OUTERMOST_I = ceil(numRows/256);		//256=III*II 
	int OUTERMOST_K = ceil(numCols/256);		// 256=KKK*KK 

    	float *A, *C;
 	void *aa=NULL,*cc=NULL;
 
	int TOTAL_I = III * II * OUTERMOST_I; 
    	int TOTAL_K = KKK * KK * OUTERMOST_K; 
    
    	long int num_elem_A = (long int)TOTAL_I * TOTAL_K; 
    	long int num_elem_C = (long int)TOTAL_I * TOTAL_I; 
    	
     	posix_memalign(&aa,ACL_ALIGNMENT,num_elem_A * sizeof(float));
    	A=(float*)aa;
    	if (A==NULL)
       	    perror("Failed malloc of matrix A");
    	posix_memalign(&cc,ACL_ALIGNMENT,num_elem_C * sizeof(float));
    	C=(float*)cc;
    	if (C==NULL)
            perror("Failed malloc of matrix C");


    	memcpy(A,arg->getValues(),num_elem_A * sizeof(float));

//#ifndef NDEBUG
	printf("\nrunning FPGA SYRK kernel \n");
//#endif    
 
	syrk(A ,C ,OUTERMOST_I, OUTERMOST_K, ctx);

    	printf("\nSyrk kernel finished!\n");


 
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(numCols, numCols, false);

	memcpy(res->getValues(),C,num_elem_C * sizeof(float));
 
    
    }
};

}
