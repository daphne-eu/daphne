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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_NUMPYMAPKERNEL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_NUMPYMAPKERNEL_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#pragma once
#include <runtime/local/kernels/MAP_EXTERNAL/NumpyTypeString.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct Ctypes_SharedMem_VoidPtr
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char * varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypes_SharedMem_VoidPtr(DTRes *& res, const DTArg * arg, const char* func, const char * varName) {
    Ctypes_SharedMem_VoidPtr<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template <typename VTRes, typename VTArg>
struct Ctypes_SharedMem_VoidPtr<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char * func, const char * varName)
    {
        PythonInterpreter::getInstance();

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_sharedMem_voidPointer");
        PyObject* pModule = PyImport_Import(pName);
        Py_XDECREF(pName);

        if (!pModule) {
            std::cerr << "Failed to import Python module!" << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        }

        PyObject* pFunc = PyObject_GetAttrString(pModule, "apply_map_function");
        Py_XDECREF(pModule);

         if (!PyCallable_Check(pFunc)) {
            std::cerr << "Function not callable!" << std::endl;
            Py_XDECREF(pFunc);
            PyGILState_Release(gstate);
            return;
        }

        //std::cout << "Dtype in C++ " << NumpyTypeString<VTArg>::value << std::end;
        std::string dtype = get_dtype_name();

        // Call the map_function with the appropriate arguments
        PyObject* pArgs = PyTuple_Pack(7,
                                        PyLong_FromVoidPtr(arg->getValuesSharedPtr().get()),
                                        PyLong_FromVoidPtr(res->getValuesSharedPtr().get()),
                                        Py_BuildValue("(ii)", arg->getNumRows(), arg->getNumCols()),
                                        Py_BuildValue("(ii)", res->getNumRows(), res->getNumCols()),
                                        PyUnicode_FromString(func),
                                        PyUnicode_FromString(varName),
                                        PyUnicode_FromString(dtype.c_str())
                                        );

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        if (!pResult) {
            PyErr_Print();
            PyGILState_Release(gstate);
        } else {
            Py_XDECREF(pResult);
        }
        Py_DECREF(pFunc);
        Py_DECREF(pArgs);

        PyGILState_Release(gstate);

    }

    static std::string get_dtype_name() {
        if (std::is_same<VTArg, float>::value) {
            return "float32";
        } else if (std::is_same<VTArg, double>::value) {
            return "float64";
        } else if (std::is_same<VTArg, int32_t>::value) {
            return "int32";
        } else if (std::is_same<VTArg, int64_t>::value) {
            return "int64";
        } else if (std::is_same<VTArg, int8_t>::value) {
            return "int8";
        } else if (std::is_same<VTArg, uint64_t>::value) {
            return "uint64";
        } else if (std::is_same<VTArg, uint8_t>::value) {
            return "uint8";
        } else {
            throw std::runtime_error("Unsupported data type!");
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_NUMPYMAPKERNEL_H