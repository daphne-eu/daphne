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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/MapExternalPL.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <catch.hpp>
#include <vector>
#include <tags.h>

#define TYPES double

/* C++ like functions in a String */

const char *squareFunc = R"(
# Square function: Computes the square of a value.
def square(x):
    return x * x
)";

const char *doubleValueFunc = R"(
# DoubleValue function: Computes the double value of a value.
def doubleValue(x):
    return x * 2
)";

const char *cubeFunc = R"(
# Cube function: Computes the cube of a value.
def cube(x):
    return x * x * x
)";

template<class DTRes, class DTArg>
void checkMap(const DTArg * arg, const DTRes * exp, const char * func) 
{
    DTRes * res = nullptr;
    mapExternalPL(res, arg, func, "x", "Python_Ctypes", nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template<template<typename VT> class DT, class VTArg, class VTRes>
void testApplyMapFunctionCtypes() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    // Create input DenseMatrix using genGivenVals
    auto input1 = genGivenVals<DTArg>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });

    auto input2 = genGivenVals<DTArg>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });

    auto input3 = genGivenVals<DTArg>(3, {
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

    checkMap(input1, squarefunc_res, squareFunc);
    checkMap(input2, doublefunc_res, doubleValueFunc);
    checkMap(input3, cubefunc_res, cubeFunc);

    DataObjectFactory::destroy(input1, input2, input3, squarefunc_res, doublefunc_res, cubefunc_res);
}

TEMPLATE_TEST_CASE("Test applyMapFunction with Alternative Kernels of other languages", "[applyMapFunction][Ctypes]", TYPES) {
    testApplyMapFunctionCtypes<DenseMatrix, TestType, TestType>();
}