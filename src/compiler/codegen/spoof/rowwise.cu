//%TMP%

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// RowType: %TYPE%
// ConstDim2: %CONST_DIM2%
// TB1: %TB1%
// VectMem: %VECT_MEM%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"
#include "Matrix.h"
#include "TempStorage.cuh"

enum RowType {
    NO_AGG_,       //no aggregation
    NO_AGG_B1_,    //no aggregation w/ matrix mult B1
    NO_AGG_CONST_, //no aggregation w/ expansion/contraction
    FULL_AGG_,     //full row/col aggregation
    ROW_AGG_,      //row aggregation (e.g., rowSums() or X %*% v)
    COL_AGG_,      //col aggregation (e.g., colSums() or t(y) %*% X)
    COL_AGG_T_,    //transposed col aggregation (e.g., t(X) %*% y)
    COL_AGG_B1_,   //col aggregation w/ matrix mult B1
    COL_AGG_B1_T_, //transposed col aggregation w/ matrix mult B1
    COL_AGG_B1R_,  //col aggregation w/ matrix mult B1 to row vector
    COL_AGG_CONST_ //col aggregation w/ expansion/contraction
};


template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
struct SpoofRowwiseOp //%HAS_TEMP_VECT%
{
	MatrixAccessor<T> a;
	MatrixAccessor<T> b[NUM_B];
	MatrixAccessor<T> c;
	T* scalars;
	uint32_t grix;
	T* avals;
	uint32_t* aix;
	uint32_t alen;

	SpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :
		        scalars(scalars), grix(grix) /*%INIT_TEMP_VECT%*/ {
		a.init(A);
		c.init(C);
		
		if(B) {
		    for(auto i = 0; i < NUM_B; ++i)
		        b[i].init(&(B[i]));
		}
	}

	__device__  __forceinline__ void exec_dense(uint32_t ai, uint32_t ci, uint32_t rix) {
//%BODY_dense%
	}

	__device__  __forceinline__ void exec_sparse(uint32_t ai, uint32_t ci, uint32_t rix, uint32_t tid, uint32_t block_dim) {
//%BODY_sparse%
	}
};


template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_DENSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,
        uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)
{
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
	spoof_op.exec_dense(rix * a->cols, rix * c->cols, rix);
};


template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__device__ void exec_sparse(Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix, uint32_t rix,
        uint32_t tid, uint32_t block_dim)
{
    SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
    spoof_op.alen = spoof_op.a.row_len(rix);
    spoof_op.aix = spoof_op.a.col_idxs(0);
    spoof_op.avals = spoof_op.a.vals(0);
    spoof_op.exec_sparse(a->row_ptr[rix], rix * c->cols, rix, tid, block_dim);
}


template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,
        uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)
{
	const uint32_t& rix = blockIdx.x;
    exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);
}


template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE_THREAD_BINS (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,
        uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size) {

    // global thread id
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gtid < bin_size) {
        // bin index (either based on thread id for short rows (bin 0) or block id
        const auto rix = bins[bin_num * a->rows + gtid];
//        if(MatrixAccessor<T>(a).row_len(rix) > 0) {
//        printf("gtid=%d < bin_size=%d; rix=%d\n", gtid, bin_size, rix);
            exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, gtid, 1);
//        }
//        else {
//            RowType row_type = %TYPE%_;
//            auto ci  = rix * c->cols;
//            if(row_type == NO_AGG_ || row_type == NO_AGG_CONST_) {
//                auto i = 0;
//                auto len = a->cols;
//                while(i < len) {
//                    c->data[ci+i++] = 0;
//                }
//            }
//            else if(row_type == ROW_AGG_)
//                c->data[rix] = 0;
//            else if(row_type == FULL_AGG_)
//                return;
//            else
//                printf("ERROR! row_type %d not handled in empty sparse row kernel\n", row_type);
//        }
    }
}


template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE_BLOCK_BINS (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars,
        T* tmp_stor, uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)
{
    // bin index (either based on thread id for short rows (bin 0) or block id
    uint32_t bix = bin_num * a->rows + blockIdx.x;
    const auto rix = bins[bix];
    exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);
}
