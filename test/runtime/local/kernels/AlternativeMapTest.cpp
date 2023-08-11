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
#include <runtime/local/datastructures/DataObjectFactory.h>
//#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/MapExternalPL.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <catch.hpp>
#include <vector>
#include <tags.h>


#define TYPES double

/* C++ like functions */
// Square function: Computes the square of a value.
template <typename VTArg, typename VTRes>
VTRes square(VTArg value) {
    return static_cast<VTRes>(value) * static_cast<VTRes>(value);
}

// DoubleValue function: Computes the double value of a value.
template <typename VTArg, typename VTRes>
VTRes doubleValue(VTArg value) {
    return static_cast<VTRes>(value) * static_cast<VTRes>(2);
}

// Cube function: Computes the cube of a value.
template <typename VTArg, typename VTRes>
VTRes cube(VTArg value) {
    return static_cast<VTRes>(value) * static_cast<VTRes>(value) * static_cast<VTRes>(value);
}

template<class DTRes, class DTArg>
void checkMap(const DTArg * arg, const DTRes * exp, void* func) {
    DTRes * res = nullptr;
    mapExternalPL(res, arg, func, "x", "PyBind", nullptr);
    if (res) {
        size_t numRows = res->getNumRows();
        size_t numCols = res->getNumCols();

        for (size_t r = 0; r < numRows; ++r) {
            for (size_t c = 0; c < numCols; ++c) {
                std::cout << res->get(r, c) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    
    //CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
    }

template<template<typename VT> class DT, class VTArg, class VTRes>
void testApplyMapFunctionPyBind() {
    std::cout << "testApplyMapFunction" << std::endl;
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    // Create input DenseMatrix using genGivenVals
    auto input = genGivenVals<DTArg>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });

    //Result Matrix: Square func
    auto squarefunc_res = genGivenVals<DTRes>(3, {
        1, 4, 9,
        16, 25, 36,
        49, 64, 81
    });

    //Result Matrix: double func
    auto doublefunc_res = genGivenVals<DTRes>(3, {
        2, 4, 6,
        8, 10, 12,
        14, 16, 18,
    });

    //Result Matrix: Cube func
    auto cubefunc_res = genGivenVals<DTRes>(3, {
        1, 8, 27,
        64, 125, 216,
        343, 512, 729
    });

    void* squareFunc = reinterpret_cast<void*>(&square<VTArg, VTRes>);
    checkMap(input, squarefunc_res, squareFunc);

    void* doubleValueFunc = reinterpret_cast<void*>(&doubleValue<VTArg, VTRes>);
    checkMap(input, doublefunc_res, doubleValueFunc);

    void* cubeFunc = reinterpret_cast<void*>(&cube<VTArg, VTRes>);
    checkMap(input, cubefunc_res, cubeFunc);
    DataObjectFactory::destroy(input, squarefunc_res, doublefunc_res, cubefunc_res);
}

TEMPLATE_TEST_CASE("Test applyMapFunction with PyBind", "[applyMapFunction][PyBind]", TYPES) {
    testApplyMapFunctionPyBind<DenseMatrix, TestType, TestType>();
}