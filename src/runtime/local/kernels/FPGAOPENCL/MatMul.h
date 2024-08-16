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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cblas.h>
#include <math.h>

#include <cstddef>
#include "gemm_interface.h"
#include "sgemv_interface.h"

namespace FPGAOPENCL {
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul {
//   static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
   static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs,transa, transb, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs, bool transa, bool transb,DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nc2 = rhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        if (nc1 != nr2)
            perror("#cols of lhs and #rows of rhs must be the same");

// Parameters of the systolic array in the bitstream. Do not change.

	//GEMM
#define II_gemm   32
#define JJ_gemm   32
#define KK_gemm   32
#define III_gemm  14
#define JJJ_gemm  16
#define KKK_gemm  16
	//GEMV
#define II_gemv   32
#define KK_gemv   32
#define III_gemv  64
#define KKK_gemv  1


//#ifndef NDEBUG
    // if (nr1%(II*III) != 0)
    //     perror("lhs #rows number must be a multiple of 448");
    // if (nc1%(JJ*JJJ) != 0 || nc1 <= 512 || nr2%(JJ*JJJ) != 0 || nc1 <= 512)
    //     perror("#cols of lhs and #rows of rhs must be a multiple of 512 (and minimum 1024)");
	// if (nc2%(KK*KKK) != 0)
    //     perror("#cols of rhs must be a multiple of 512");
//#endif       

//	printf("\ntest MatMul f32 \n");
// Parameters of the systolic array in the bitstream. Do not change.

//#define TYPE float

#define ACL_ALIGNMENT 64
    int OUTERMOST_I; //= ceil(nr1/448);
    int OUTERMOST_J; //= ceil(nc2/512);
    int OUTERMOST_K; //= ceil(nc1/512);

    float *A, *B, *C;
    void *aa=NULL,*bb=NULL,*cc=NULL;
 
 
    int TOTAL_I; //= III * II * OUTERMOST_I;
    int TOTAL_J; //= JJJ * JJ * OUTERMOST_J;
    int TOTAL_K; //= KKK * KK * OUTERMOST_K;
    
    long int num_elem_A; // = (long int)TOTAL_I*TOTAL_K;
    long int num_elem_B; // = (long int)TOTAL_K*TOTAL_J;
    long int num_elem_C; // = (long int)TOTAL_I*TOTAL_J;
    
#ifndef NDEBUG
    printf("\nA rows %ld\n",nr1);
    printf("\nA cols %ld\n",nc1);
    printf("\nX rows %ld\n",nr2);
    printf("\nX cols %ld\n",nc2);
#endif 

    
    if(nc2==1)//gemv kernel
    {
#ifndef NDEBUG
	printf("\nrunning GEMV kernel \n");
#endif    
        OUTERMOST_I = ceil(nr1/2048);
    	OUTERMOST_K = ceil(nc1/32);
 
    	TOTAL_I = III_gemv * II_gemv * OUTERMOST_I;
    	TOTAL_K = KKK_gemv * KK_gemv * OUTERMOST_K;
    
    	num_elem_A = (long int)TOTAL_I*TOTAL_K;
    	num_elem_B = (long int)TOTAL_K;
    	num_elem_C = (long int)TOTAL_I;
    
    	posix_memalign(&aa,ACL_ALIGNMENT,num_elem_A * sizeof(float));
    	A=(float*)aa;
    	if (A==NULL)
       	    perror("Failed malloc of matrix A");
    	posix_memalign(&bb,ACL_ALIGNMENT,num_elem_B * sizeof(float));
    	B=(float*)bb;
    	if (B==NULL)
           perror("Failed malloc of matrix B");
    	posix_memalign(&cc,ACL_ALIGNMENT,num_elem_C * sizeof(float));
    	C=(float*)cc;
    	if (C==NULL)
            perror("Failed malloc of matrix C");

    	memcpy(A,lhs->getValues(),num_elem_A * sizeof(float));//sizeof(lhs));
    	memcpy(B,rhs->getValues(),num_elem_B * sizeof(float));//sizeof(rhs));
    	
	sgemv(A, B, C, OUTERMOST_I, OUTERMOST_K, ctx);

   	 if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);
   
    	memcpy(res->getValues(),C,num_elem_C * sizeof(float));//sizeof(C)
 
    }
    else //gemm kernel
    {   
#ifndef NDEBUG
	printf("\nrunning GEMM kernel \n");
#endif    
    	OUTERMOST_I = ceil(nr1/448);
    	OUTERMOST_J = ceil(nc2/512);
    	OUTERMOST_K = ceil(nc1/512);
 
    	TOTAL_I = III_gemm * II_gemm * OUTERMOST_I;
    	TOTAL_J = JJJ_gemm * JJ_gemm * OUTERMOST_J;
    	TOTAL_K = KKK_gemm * KK_gemm * OUTERMOST_K;
    
    	num_elem_A = (long int)TOTAL_I*TOTAL_K;
    	num_elem_B = (long int)TOTAL_K*TOTAL_J;
    	num_elem_C = (long int)TOTAL_I*TOTAL_J;
    
    	posix_memalign(&aa,ACL_ALIGNMENT,num_elem_A * sizeof(float));
    	A=(float*)aa;
    	if (A==NULL)
            perror("Failed malloc of matrix A");
    	posix_memalign(&bb,ACL_ALIGNMENT,num_elem_B * sizeof(float));
    	B=(float*)bb;
    	if (B==NULL)
            perror("Failed malloc of matrix B");
    	posix_memalign(&cc,ACL_ALIGNMENT,num_elem_C * sizeof(float));
    	C=(float*)cc;
    	if (C==NULL)
            perror("Failed malloc of matrix C");

    	memcpy(A,lhs->getValues(),num_elem_A * sizeof(float));
    	memcpy(B,rhs->getValues(),num_elem_B * sizeof(float));
     
    	sgemm(A, B, C, OUTERMOST_I, OUTERMOST_J, OUTERMOST_K, ctx);
 
    	if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);
   
    	memcpy(res->getValues(),C,num_elem_C * sizeof(float));
   }

   }
};
/* TODO
template<>
struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nc2 = rhs->getNumCols();
#ifndef NDEBUG
        const size_t nr2 = rhs->getNumRows();
        if (nc1 != nr2)
            perror("#cols of lhs and #rows of rhs must be the same");
#endif 
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            res->set(0, 0, cblas_ddot(nc1, lhs->getValues(), 1, rhs->getValues(), rhs->getRowSkip()));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nr1, nc1, 1, lhs->getValues(),
                lhs->getRowSkip(), rhs->getValues(), rhs->getRowSkip(), 0,
                res->getValues(), res->getRowSkip());
        else                     // Matrix-Matrix
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr1, nc2, nc1,
                1, lhs->getValues(), lhs->getRowSkip(), rhs->getValues(),
                rhs->getRowSkip(), 0, res->getValues(), res->getRowSkip());
    }
};
*/
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H
