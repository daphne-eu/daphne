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
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_sharedMem_address.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_sharedMem_Pointer.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_sharedMem_voidPointer.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_csv.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_binaryData.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_copy.h>
#include <runtime/local/kernels/MAP_EXTERNAL/CtypesMapKernel_SysArg.h>
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

            if (strcmp(plName, "Python_Ctypes_sharedMem_voidPointer") == 0)
            {
                applyCTypes_SharedMem_VoidPtr(res, arg, func, varName);
            }
            else if (strcmp(plName, "Python_Ctypes_sharedMem_address") == 0)
            {
                applyCTypesKernel_SharedMem_Adress(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_sharedMem_Pointer") == 0)
            {
                applyCTypesMapKernel_SharedMem_Pointer(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_copy") == 0)
            {
                applyCTypesMapKernel_copy(res, arg, func, varName);
            }
            else if(strcmp(plName, "Python_Ctypes_csv") == 0)
            {
                applyCTypesMapKernel_csv(res, arg, func, varName);
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
                std::cerr << "Programming language " << plName << " can't be used" <<std::endl;
                throw std::runtime_error("Programming Language can't be used");
            }

        }
        else
        {
            
            throw std::runtime_error("Programming Language field is NULL");
        }
    }

    static void applyCTypesKernel_SharedMem_Adress(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesKernel_SharedMem_Adress(res, arg, func, varName);
    }

    static void applyCTypesMapKernel_SharedMem_Pointer(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypesMapKernel_SharedMem_Pointer(res, arg, func, varName);
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

    static void applyCTypes_SharedMem_VoidPtr(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
        ctypes_SharedMem_VoidPtr(res, arg, func, varName);
    }

};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAPEXTERNALPL_H