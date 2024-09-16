#include <immintrin.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <unistd.h>

#include "runtime/local/datastructures/DataObjectFactory.h"

#include <iostream>
#include <stdexcept>

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

class DaphneContext;

// Horizontal sum of [4 x double] __m256d
inline double hsum_double_avx2(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

extern "C" {

void spmv_simd_parallel_omp(DenseMatrix<double> *&res,
                            const CSRMatrix<double> *lhs,
                            const DenseMatrix<double> *rhs, bool transa,
                            bool transb, DaphneContext *ctx) {
    LIKWID_MARKER_INIT;
    const size_t nr_lhs = lhs->getNumRows();
    [[maybe_unused]] const size_t nc_lhs = lhs->getNumCols();

    [[maybe_unused]] const size_t nr_rhs = rhs->getNumRows();
    const size_t nc_rhs = rhs->getNumCols();

    if (nc_lhs != nr_rhs) {
        throw std::runtime_error(
            "Gemv - #cols of mat and #rows of vec must be the same");
    }

    if (res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<double>>(nr_lhs, nc_rhs,
                                                             false);

    const auto *valuesRhs = rhs->getValues();
    auto *valuesRes = res->getValues();
    memset(valuesRes, double(0), sizeof(double) * nr_lhs * nc_rhs);

    auto *row_offsets = lhs->getRowOffsets();
    auto *values = lhs->getValues();
    auto *col_idx = lhs->getColIdxs();

#pragma omp parallel
    {
        LIKWID_MARKER_START("spmv_simd_parallel_omp");
#pragma omp for
        for (size_t row = 0; row < nr_lhs; ++row) {
            double row_sum = 0;
            // Initialize [4 x double] row-accumulator
            __m256d row_acc = _mm256_setzero_pd();
            // Iterate over non-zero elements in row
            auto values_in_row = row_offsets[row + 1] - row_offsets[row];
            int rounds = values_in_row / 4;
            for (int i = 0; i < rounds; ++i) {
                int idx = row_offsets[row] + i * 4;
                // Load doubles from LHS matrix
                __m256d mat_v = _mm256_loadu_pd(&values[idx]);
                // Load RHS column indices
                __m256i col_idxs =
                    _mm256_loadu_si256((const __m256i *)&col_idx[idx]);
                // Gather values from RHS vector
                __m256d vec_v = _mm256_i64gather_pd(valuesRhs, col_idxs, 8);
                // Multiply and add to accumulator
                row_acc = _mm256_fmadd_pd(mat_v, vec_v, row_acc);
            }
            // Horizontal sum of accumulator
            row_sum = hsum_double_avx2(row_acc);
            // Handle remaining elements
            for (auto i = row_offsets[row] + rounds * 4;
                 i < row_offsets[row + 1]; ++i) {
                row_sum += values[i] * valuesRhs[col_idx[i]];
            }
            // Store result
            valuesRes[row] = row_sum;
        }

        LIKWID_MARKER_STOP("spmv_simd_parallel_omp");
    }
    LIKWID_MARKER_CLOSE;
}
}
