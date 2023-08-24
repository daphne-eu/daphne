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
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel.h>
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel_sharedPointer.h>
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel_csv.h>
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel_binaryData.h>
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel_copy.h>
#include <runtime/local/kernels/MAP_CTYPES/CtypesMapKernel_SysArg.h>
#include <runtime/local/kernels/MAP_NUMPY/NumpyMapKernel.h>
#include <memory>
#include <algorithm>
#include <cassert>

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
void mapExternalPL(DTRes *& res, const DTArg * arg, const char * func, const char* varName, const char* plName, DCTX(ctx)) {
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
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char * func, const char * varName, const char * plName, DCTX(ctx)) {
        
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        
        if(plName != NULL)
        {

            if (strcmp(plName, "Python_Numpy") == 0)
            {
                applyNumpyKernel(res, arg, func, varName);
            }
            else if (strcmp(plName, "Python_Ctypes") == 0)
            {
                applyCTypesKernel(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_copy") == 0)
            {
                applyCTypesMapKernel_copy(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_csv") == 0)
            {
                applyCTypesMapKernel_csv(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_sharedPointer") == 0)
            {
                applyCTypesMapKernel_SharedPointer(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_binaryData") == 0)
            {
                applyCTypesMapKernel_binaryData(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_SysArg") == 0)
            {
                applyCtypesMapKernel_SysArg(res, arg, func, varName);
            }
            else
            {
                throw std::runtime_error("Programming Language can't be used");
            }

        }
        else
        {
            throw std::runtime_error("Programming Language field is NULL");
        }
    }

    static void applyCTypesKernel(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel(res, arg, func, varName);
    }

    static void applyCTypesMapKernel_SharedPointer(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_SharedPointer(res, arg, func, varName);
    }

    static void applyCTypesMapKernel_csv(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_csv(res, arg, func, varName);
    }

    static void applyCTypesMapKernel_binaryData(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_binaryData(res, arg, func, varName);
    }

    static void applyCTypesMapKernel_copy(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_copy(res, arg, func, varName);
    }

    static void applyCtypesMapKernel_SysArg(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_SysArg(res, arg, func, varName);
    }

    static void applyNumpyKernel(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        numpyMapKernel(res, arg, func, varName);
    }

};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAPEXTERNALPL_H