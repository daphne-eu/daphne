/*
 * Copyright 2024 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <string>

#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct TypeOfObj {
    static void apply(char *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void typeOfObj(char *&res, const DTArg *arg, DCTX(ctx)) {
    TypeOfObj<DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TypeOfObj<DenseMatrix<VT>> {
    static void apply(char *&res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const std::string typeName = std::string("DenseMatrix(") + std::to_string(arg->getNumRows()) + "x" +
                                     std::to_string(arg->getNumCols()) + ", " + ValueTypeUtils::cppNameFor<VT> + ")";
        if (res == nullptr)
            res = new char[typeName.size() + 1];
        std::memcpy(res, typeName.c_str(), typeName.size() + 1);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TypeOfObj<CSRMatrix<VT>> {
    static void apply(char *&res, const CSRMatrix<VT> *arg, DCTX(ctx)) {
        const std::string typeName = std::string("CSRMatrix(") + std::to_string(arg->getNumRows()) + "x" +
                                     std::to_string(arg->getNumCols()) + ", " + ValueTypeUtils::cppNameFor<VT> + ")";
        if (res == nullptr)
            res = new char[typeName.size() + 1];
        std::memcpy(res, typeName.c_str(), typeName.size() + 1);
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct TypeOfObj<Frame> {
    static void apply(char *&res, const Frame *arg, DCTX(ctx)) {
        std::string typeName =
            std::string("Frame(") + std::to_string(arg->getNumRows()) + "x" + std::to_string(arg->getNumCols()) + ", [";
        const std::string *labels = arg->getLabels();
        for (size_t i = 0; i < arg->getNumCols(); i++) {
            typeName += labels[i] + ":" + ValueTypeUtils::cppNameForCode(arg->getColumnType(i));
            if (i < arg->getNumCols() - 1) {
                typeName += ", ";
            }
        }
        typeName += "])";
        if (res == nullptr)
            res = new char[typeName.size() + 1];
        std::memcpy(res, typeName.c_str(), typeName.size() + 1);
    }
};