#include <runtime/local/datastructures/DenseMatrix.h>

#include <immintrin.h> // for the SIMD-enabled kernel
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <omp.h>

class DaphneContext;


extern "C" {

    #define SIMD_VECTOR_ELEMENT_COUNT 8

    #ifndef MAX_THREADS
    constexpr size_t max_threads = 4;
    #else
    constexpr size_t max_threads = MAX_THREADS;
    #endif

    void add_simd(__m512i &inout, __m512i &in) {
        inout = _mm512_add_epi64(inout, in);
    }

    #pragma omp declare reduction(                             \
                    simd_add :                     \
                    __m512i :   \
                    add_simd(omp_out, omp_in)      \
                    )                             \
            initializer (omp_priv=omp_orig)



    void aggScalar(
        int64_t * res,
        const DenseMatrix<int64_t> * arg,
        DaphneContext * ctx
    ) {
        std::cerr << "hello from mySumSeq()" << std::endl;
        const int64_t * valuesArg = arg->getValues();
        *res = 0;
        for(size_t r = 0; r < arg->getNumRows(); r++) {
            for(size_t c = 0; c < arg->getNumCols(); c++)
                *res += valuesArg[c];
            valuesArg += arg->getRowSkip();
        }
    }

    void aggSIMDLoad(
        int64_t * res,
        const DenseMatrix<int64_t> *arg,
        DaphneContext * ctx
    ){
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % SIMD_VECTOR_ELEMENT_COUNT)
            throw std::runtime_error(
                "for simplicity, the number of cells must be "
                "a multiple of SIMD_VECTOR_ELEMENT_COUNT"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must not be "
                "a column segment of another matrix"
            );

        const int64_t * valuesArg = arg->getValues();
        __m512i result_vec = _mm512_set1_epi64(0);
        
        #pragma omp parallel for schedule(static) num_threads(max_threads) reduction(simd_add: result_vec)
        for(size_t i = 0; i < numCells; i += SIMD_VECTOR_ELEMENT_COUNT){
            __m512i loaded = _mm512_loadu_epi64(&valuesArg[i]);
            result_vec = _mm512_add_epi64(loaded, result_vec);
        }

        *res = _mm512_reduce_add_epi64(result_vec);
    }

    void aggSIMDGather(
        int64_t * res,
        const DenseMatrix<int64_t> *arg,
        DaphneContext * ctx
    ){
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % (SIMD_VECTOR_ELEMENT_COUNT * max_threads)){
            throw std::runtime_error(
                "for simplicity, the number of cells must be "
                "a multiple of SIMD_VECTOR_ELEMENT_COUNT and Thread Count"
            );
        }
        if(arg->getNumCols() != arg->getRowSkip()){
            throw std::runtime_error(
                "for simplicity, the argument must not be "
                "a column segment of another matrix"
            );
        }

        const int64_t * valuesArg = arg->getValues(); 
        __m512i result_vec = _mm512_set1_epi64(0);
        const size_t partition_size = numCells / max_threads; // tuple per thread
        const size_t step_size = partition_size / SIMD_VECTOR_ELEMENT_COUNT; 
        __m512i offset = _mm512_set_epi64(step_size * 7, step_size * 6, step_size * 5, step_size * 4, step_size * 3, step_size * 2, step_size * 1, step_size * 0); // offset of different pages
        #pragma omp parallel num_threads(max_threads) reduction(simd_add: result_vec)    
        {
            const size_t thread_id = omp_get_thread_num();
            const size_t thread_start_index = partition_size * thread_id;
            const size_t thread_end_index = thread_start_index + step_size;
        
            for(size_t pos = thread_start_index; pos < thread_end_index; pos ++) {
                __m512i loaded = _mm512_i64gather_epi64(offset, &valuesArg[pos], 8);
                result_vec = _mm512_add_epi64(loaded, result_vec);
            }
        }
        *res = _mm512_reduce_add_epi64(result_vec);
    }




}