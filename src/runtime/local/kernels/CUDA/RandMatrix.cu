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

#include "RandMatrix.h"

namespace CUDA {
    template<typename T>
    void launch_gen_rand(curandGenerator_t& prng, T* buf, size_t count);

    template<>
    void launch_gen_rand<float>(curandGenerator_t& prng, float* buf, size_t count) {
        CHECK_CURAND(curandGenerateUniform(prng, buf, count));
    }

    template<>
    void launch_gen_rand<double>(curandGenerator_t& prng, double* buf, size_t count) {
        CHECK_CURAND(curandGenerateUniformDouble(prng, buf, count));
    }

    // ToDo: CUDA rng for integers ?
    template<>
    void launch_gen_rand<int64_t>(curandGenerator_t& prng, int64_t* buf, size_t count) {
        CHECK_CURAND(curandGenerateUniformDouble(prng, reinterpret_cast<double*>(buf), count));
    }

    template<>
    void launch_gen_rand<uint8_t>(curandGenerator_t& prng, uint8_t* buf, size_t count) {
        CHECK_CURAND(curandGenerateUniform(prng, reinterpret_cast<float*>(buf), count));
    }


    template<typename VT>
    void RandMatrix<DenseMatrix<VT>, VT>::apply(DenseMatrix<VT> *&res, size_t numRows, size_t numCols, VT min, VT max,
                                                double sparsity, int64_t seed, DCTX(dctx)) {
        auto ctx = dctx->getCUDAContext(0);
        assert(numRows > 0 && "numRows must be > 0");
        assert(numCols > 0 && "numCols must be > 0");
        assert(min <= max && "min must be <= max");
        assert((min != 0 || max != 0) &&
               "min and max must not both be zero, consider setting sparsity to zero instead");
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false, ALLOCATION_TYPE::CUDA_ALLOC);

        auto prngGPU = ctx->getRandomGenerator();
        CHECK_CURAND(curandSetStream(prngGPU, ctx->getCuRandStream()));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));

        printf("Generating random numbers on GPU...\n\n");

        launch_gen_rand(prngGPU, res->getValuesCUDA(), numRows * numCols);

        CHECK_CUDART(cudaStreamSynchronize(ctx->getCuRandStream()));

    }
    template struct RandMatrix<DenseMatrix<double>, double>;
    template struct RandMatrix<DenseMatrix<float>, float>;
    template struct RandMatrix<DenseMatrix<int64_t>, int64_t>;
    template struct RandMatrix<DenseMatrix<uint8_t>, uint8_t>;
}