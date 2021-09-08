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

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/vectorized/MTWrapper.h>

#include <tags.h>
#include <catch.hpp>
#include <cstdint>

#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, float //TODO uint32_t

template<class VT>
void funAdd(DenseMatrix<VT>* res, DenseMatrix<VT>* lhs, DenseMatrix<VT>* rhs) {
    ewBinaryMat(BinaryOpCode::ADD, res, lhs, rhs, nullptr);
}

template<class VT>
void funMul(DenseMatrix<VT>* res, DenseMatrix<VT>* lhs, DenseMatrix<VT>* rhs) {
    ewBinaryMat(BinaryOpCode::MUL, res, lhs, rhs, nullptr);
}

TEMPLATE_PRODUCT_TEST_CASE("Multi-threaded X+Y", TAG_VECTORIZED, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *m1 = nullptr, *m2 = nullptr;
    randMatrix<DT, VT>(m1, 1234, 10, 0.0, 1.0, 1.0, 7, nullptr);
    randMatrix<DT, VT>(m2, 1234, 10, 0.0, 1.0, 1.0, 3, nullptr);

    DT *r1 = nullptr, *r2 = nullptr;
    ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, r1, m1, m2, nullptr); //single-threaded
    MTWrapper<VT>* wrapper = new MTWrapper<VT>(4);

    
    wrapper->execute(&funAdd, r2, m1, m2, false, "SCH_STATIC"); //multi-threaded
    //FIXME missing util function for templated approx checks
    CHECK(Approx(*(r1->getValues())).epsilon(1e-6) == *(r2->getValues()));

    wrapper->execute(&funAdd, r2, m1, m2, false, "SCH_GSS"); //multi-threaded
    //FIXME missing util function for templated approx checks
    CHECK(Approx(*(r1->getValues())).epsilon(1e-6) == *(r2->getValues()));

    wrapper->execute(&funAdd, r2, m1, m2, false, "SCH_TFSS"); //multi-threaded
    //FIXME missing util function for templated approx checks
    CHECK(Approx(*(r1->getValues())).epsilon(1e-6) == *(r2->getValues()));


    delete wrapper;
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(r1);
    DataObjectFactory::destroy(r2);
}

TEMPLATE_PRODUCT_TEST_CASE("Multi-threaded X*Y", TAG_VECTORIZED, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *m1 = nullptr, *m2 = nullptr;
    randMatrix<DT, VT>(m1, 1234, 10, 0.0, 1.0, 1.0, 7, nullptr);
    randMatrix<DT, VT>(m2, 1234, 10, 0.0, 1.0, 1.0, 3, nullptr);

    DT *r1 = nullptr, *r2 = nullptr;
    ewBinaryMat<DT, DT, DT>(BinaryOpCode::MUL, r1, m1, m2, nullptr); //single-threaded
    MTWrapper<VT>* wrapper = new MTWrapper<VT>(4);
    
    wrapper->execute(&funMul, r2, m1, m2, false); //multi-threaded

    //FIXME missing util function for templated approx checks
    CHECK(Approx(*(r1->getValues())).epsilon(1e-6) == *(r2->getValues()));

    delete wrapper;
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(r1);
    DataObjectFactory::destroy(r2);
}
