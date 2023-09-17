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

#define TYPES double, float, int64_t, int32_t, int8_t, uint64_t, uint32_t, uint8_t

const char *squareFunc = R"(
# Square function: Computes the square of a value.
def square(x):
    return x * x
)";

const char *squareFuncLambda = R"(x**2)";

const char *doubleValueFunc = R"(
# DoubleValue function: Computes the double value of a value.
def doubleValue(x):
    return x * 2
)";

const char *doubleValueFuncLambda = R"(x*2)";

const char *cubeFunc = R"(
# Cube function: Computes the cube of a value.
def cube(x):
    return x * x * x
)";

const char *absFunc = R"(
# Absolute function: Computes the absolute value of a number.
def absolute(x):
    return abs(x)
)";

const char *roundFunc = R"(
# Round function: Rounds the number to the nearest integer.
def roundNum(x):
    return round(x)
)";

template<class DTRes, class DTArg>
void checkMap(const DTArg * arg, const DTRes * exp, const char * func, const char * plName) 
{
    DTRes * res = nullptr;
    mapExternalPL(res, arg, func, "x", plName, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template<class DTRes, class DTArg>
void checkMapForAllPL(const DTArg * arg, const DTRes * exp, const char * func)
{
    checkMap(arg, exp, func, "Python_Shared_Mem");
    checkMap(arg, exp, func, "Python_CopyInMemory");
    checkMap(arg, exp, func, "Python_BinaryFile");
    checkMap(arg, exp, func, "Python_CsvFile");
    checkMap(arg, exp, func, "Python_DirectExec");
}

template<template<typename VT> class DT, class VTArg, class VTRes>
void testApplyMapFunctionCtypes() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    // Create input DenseMatrix using genGivenVals
    auto input = genGivenVals<DTArg>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });

    // Result Matrix: Square func
    auto squarefunc_res = genGivenVals<DTRes>(3, {
        1, 4, 9,
        16, 25, 36,
        49, 64, 81
    });

    // Result Matrix: double func
    auto doublefunc_res = genGivenVals<DTRes>(3, {
        2, 4, 6,
        8, 10, 12,
        14, 16, 18,
    });

    checkMapForAllPL(input, squarefunc_res, squareFunc);
    checkMapForAllPL(input, squarefunc_res, squareFuncLambda);
    checkMapForAllPL(input, doublefunc_res, doubleValueFunc);
    checkMapForAllPL(input, doublefunc_res, doubleValueFuncLambda);

    // For integer tests
    if constexpr (std::is_integral_v<VTArg>) {
        auto absfunc_res = genGivenVals<DTRes>(3, {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        });
        checkMapForAllPL(input, absfunc_res, absFunc);
    }

    // For float tests
    if constexpr (std::is_floating_point_v<VTArg>) {
        auto roundfunc_res = genGivenVals<DTRes>(3, {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        });
        checkMapForAllPL(input, roundfunc_res, roundFunc);
        DataObjectFactory::destroy(roundfunc_res);
    }

    //For Double tests
    if constexpr (std::is_same_v<VTArg, double>) {
        // Result Matrix: Cube func
        auto cubefunc_res = genGivenVals<DTRes>(3, {
            1, 8, 27,
            64, 125, 216,
            343, 512, 729
        });

        checkMapForAllPL(input, cubefunc_res, cubeFunc);
        DataObjectFactory::destroy(cubefunc_res);
    }

    DataObjectFactory::destroy(input, squarefunc_res, doublefunc_res);
}

TEMPLATE_TEST_CASE("Test applyMapFunction with Alternative Kernels of other languages", "[applyMapFunction][Ctypes]", TYPES) {
    testApplyMapFunctionCtypes<DenseMatrix, TestType, TestType>();
}