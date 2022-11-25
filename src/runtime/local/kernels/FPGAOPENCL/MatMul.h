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

#include <cassert>
#include <cstddef>
#include "gemm_interface.h"

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
#ifndef NDEBUG
        const size_t nr2 = rhs->getNumRows();
        assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");        
#endif
//	printf("\ntest MatMul f32 \n");
// Parameters of the systolic array in the bitstream. Do not change.

#define II   32
#define JJ   32
#define KK   32
#define III  14
#define JJJ  16
#define KKK  16

#ifndef NDEBUG
	assert((nr1%(II*III)==0) && "lhs #rows number must be a multiple of 448");        
	assert((nc1%(JJ*JJJ)==0 && nc1>512 && nr2%(JJ*JJJ)==0 && nc1>512) && "#cols of lhs and #rows of rhs must be a multiple of 512 (and minimum 1024)");        
	assert((nc2%(KK*KKK)==0) && "#cols of rhs must be a multiple of 512");        
#endif       
// Testing purpose only: help define the sizes of test inputs
// Can be arbitrarily set.
// matrix a: 10K * 2K
// matrix b: 2K * 8K

//#define OUTERMOST_I 1//32
//#define OUTERMOST_J 1//32
//#define OUTERMOST_K 2//4

//#define TYPE float

#define ACL_ALIGNMENT 64
//void *acl_aligned_malloc(size_t size) {
//    void *result = NULL;
//    posix_memalign(&result, ACL_ALIGNMENT, size);
//    return result;
//}
    const int OUTERMOST_I = ceil(nr1/448);
    const int OUTERMOST_J = ceil(nc2/512);
    const int OUTERMOST_K = ceil(nc1/512);

    float *A, *B, *C;
    void *aa=NULL,*bb=NULL,*cc=NULL;
    const int TOTAL_I = III * II * OUTERMOST_I;
    const int TOTAL_J = JJJ * JJ * OUTERMOST_J;
    const int TOTAL_K = KKK * KK * OUTERMOST_K;
    
    long int num_elem_A = (long int)TOTAL_I*TOTAL_K;
    long int num_elem_B = (long int)TOTAL_K*TOTAL_J;
    long int num_elem_C = (long int)TOTAL_I*TOTAL_J;
    
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


    // printf("\nbefore memcpy()\n");
   
    memcpy(A,lhs->getValues(),num_elem_A * sizeof(float));//sizeof(lhs));
    memcpy(B,rhs->getValues(),num_elem_B * sizeof(float));//sizeof(rhs));
     
    //printf("\nA values %f\n",*A);
    //printf("\nB values %f\n",*B);
    sgemm(A, B, C, OUTERMOST_I, OUTERMOST_J, OUTERMOST_K, ctx);
 
  //  printf("\nC values %f\n",*C);
 
    if(res == nullptr)
       res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);


    //printf("\nres: %p\n", res);
    //printf("\nres->getValues(): %p\n", res->getValues());
    memcpy(res->getValues(),C,num_elem_C * sizeof(float));//sizeof(C)
   // printf("\nres memcpy2\n");

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
        assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");
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
