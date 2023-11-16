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

#include <run_tests.h>

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/vectorized/MTWrapper.h>

#include <tags.h>
#include <catch.hpp>

#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, float //TODO uint32_t

template<class DT>
void funAdd(DT*** outputs, Structure** inputs, DCTX(ctx)) {
    ewBinaryMat(BinaryOpCode::ADD,
        *outputs[0],
        reinterpret_cast<DT*>(inputs[0]),
        reinterpret_cast<DT*>(inputs[1]),
        ctx);
}

template<class DT>
void funMul(DT*** outputs, Structure** inputs, DCTX(ctx)) {
    ewBinaryMat(BinaryOpCode::MUL,
        *outputs[0],
        reinterpret_cast<DT*>(inputs[0]),
        reinterpret_cast<DT*>(inputs[1]),
        ctx);
}

TEMPLATE_PRODUCT_TEST_CASE("Multi-threaded-scheduling", TAG_VECTORIZED, (DATA_TYPES), (VALUE_TYPES)){
    using DT = TestType;
    using VT = typename DT::VT;
    auto dctx = setupContextAndLogger();
    dctx->config.taskPartitioningScheme = GSS;
    dctx->config.minimumTaskSize = 50;

    DT *m1 = nullptr, *m2 = nullptr;
    randMatrix<DT, VT>(m1, 1234, 10, 0.0, 1.0, 1.0, 7, dctx.get());
    randMatrix<DT, VT>(m2, 1234, 10, 0.0, 1.0, 1.0, 3, dctx.get());

    DT *r1 = nullptr, *r2 = nullptr;
    ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, r1, m1, m2, dctx.get()); //single-threaded

    auto wrapper = std::make_unique<MTWrapper<DT>>(1, dctx.get());
    DT **outputs[] = {&r2};
    bool isScalar[] = {false, false};
    Structure *inputs[] = {m1, m2};
    int64_t outRows[] = {1234};
    int64_t outCols[] = {10};
    VectorSplit splits[] = {VectorSplit::ROWS, VectorSplit::ROWS};
    VectorCombine combines[] = {VectorCombine::ROWS};

    std::vector<std::function<void(DT ***, Structure **, DCTX(ctx))>> funcs;
    funcs.push_back(std::function<void(DT***, Structure**, DCTX(ctx))>(reinterpret_cast<void (*)(DT***, Structure **, 
            DCTX(ctx))>(reinterpret_cast<void*>(&funAdd<DT>))));
    wrapper->executeCpuQueues(funcs, outputs, isScalar, inputs, 2, 1, outRows, outCols, splits, combines, dctx.get(), false);

    CHECK(checkEqApprox(r1, r2, 1e-6, dctx.get()));

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(r1);
    DataObjectFactory::destroy(r2);
}

TEMPLATE_PRODUCT_TEST_CASE("Multi-threaded X+Y", TAG_VECTORIZED, (DATA_TYPES), (VALUE_TYPES)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    auto dctx = setupContextAndLogger();

    DT *m1 = nullptr, *m2 = nullptr;
    randMatrix<DT, VT>(m1, 1234, 10, 0.0, 1.0, 1.0, 7, dctx.get());
    randMatrix<DT, VT>(m2, 1234, 10, 0.0, 1.0, 1.0, 3, dctx.get());

    DT *r1 = nullptr, *r2 = nullptr;
    ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, r1, m1, m2, dctx.get()); //single-threaded

    auto wrapper = std::make_unique<MTWrapper<DT>>(1, dctx.get());
    DT **outputs[] = {&r2};
    bool isScalar[] = {false, false};
    Structure *inputs[] = {m1, m2};
    int64_t outRows[] = {1234};
    int64_t outCols[] = {10};
    VectorSplit splits[] = {VectorSplit::ROWS, VectorSplit::ROWS};
    VectorCombine combines[] = {VectorCombine::ROWS};

    std::vector<std::function<void(DT ***, Structure **, DCTX(ctx))>> funcs;
    funcs.push_back(std::function<void(DT***, Structure**, DCTX(ctx))>(reinterpret_cast<void (*)(DT***, Structure **,
            DCTX(ctx))>(reinterpret_cast<void*>(&funAdd<DT>))));
    wrapper->executeCpuQueues(funcs, outputs, isScalar, inputs, 2, 1, outRows, outCols, splits, combines, dctx.get(), false);

    CHECK(checkEqApprox(r1, r2, 1e-6, dctx.get()));

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(r1);
    DataObjectFactory::destroy(r2);
}

TEMPLATE_PRODUCT_TEST_CASE("Multi-threaded X*Y", TAG_VECTORIZED, (DATA_TYPES), (VALUE_TYPES)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    auto dctx = setupContextAndLogger();

    DT *m1 = nullptr, *m2 = nullptr;
    randMatrix<DT, VT>(m1, 1234, 10, 0.0, 1.0, 1.0, 7, dctx.get());
    randMatrix<DT, VT>(m2, 1234, 10, 0.0, 1.0, 1.0, 3, dctx.get());

    DT *r1 = nullptr, *r2 = nullptr;
    ewBinaryMat<DT, DT, DT>(BinaryOpCode::MUL, r1, m1, m2, dctx.get()); //single-threaded

    auto wrapper = std::make_unique<MTWrapper<DT>>(1, dctx.get());
    DT **outputs[] = {&r2};
    bool isScalar[] = {false, false};
    Structure *inputs[] = {m1, m2};
    int64_t outRows[] = {1234};
    int64_t outCols[] = {10};
    VectorSplit splits[] = {VectorSplit::ROWS, VectorSplit::ROWS};
    VectorCombine combines[] = {VectorCombine::ROWS};

    std::vector<std::function<void(DT ***, Structure **, DCTX(ctx))>> funcs;
    funcs.push_back(std::function<void(DT***, Structure**, DCTX(ctx))>(reinterpret_cast<void (*)(DT***, Structure **,
            DCTX(ctx))>(reinterpret_cast<void*>(&funMul<DT>))));
    wrapper->executeCpuQueues(funcs, outputs, isScalar, inputs, 2, 1, outRows, outCols, splits, combines, dctx.get(), false);

    CHECK(checkEqApprox(r1, r2, 1e-6, dctx.get()));

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(r1);
    DataObjectFactory::destroy(r2);
}
