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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAPEXTERNALPL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAPEXTERNALPL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/MAP_PYBIND/PyBindMapKernel.h>
#include <memory>
#include <algorithm>
#include <cassert>

// ****************************************************************************
// Workaround Methods
// ****************************************************************************
/*
template <class DT>
struct IsDenseMatrix : std::false_type {};

template <typename VT>
struct IsDenseMatrix<DenseMatrix<VT>> : std::true_type {};

template <typename T>
struct IsPointer : std::false_type {};

template <typename T>
struct IsPointer<T*> : std::true_type {};
*/

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct MapExternalPL {
    static void apply(DTRes *& res, const DTArg * arg, void* func, const char* varName, const char* plName, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void mapExternalPL(DTRes *& res, const DTArg * arg, void* func, const char* varName, const char* plName, DCTX(ctx)) {
    MapExternalPL<DTRes,DTArg>::apply(res, arg, func, varName, plName, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct MapExternalPL<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, void* func, const char* varName, const char* plName, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        
        if(plName == NULL){
            throw std::runtime_error("Programming Language field is NULL");
        }
        else if (strcmp(plName, "PyBind") == 0)
        {
            std::cout << "Starting PyBind Kernel" << std::endl;
            applyPyBindKernel(res, arg, func, varName);
        }
        throw std::runtime_error("Programming Language can't be used");
    }

    static void applyPyBindKernel(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, void* func, const char* varName) {
        PyBindMapKernel<VTRes, VTArg>::applyMapFunction(res, arg, func);
    }

};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAPEXTERNALPL_H