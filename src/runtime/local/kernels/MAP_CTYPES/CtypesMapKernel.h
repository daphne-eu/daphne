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

#pragma once
#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct CtypesMapKernel
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypesMapKernel(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

struct PyDeleter {
    void operator()(PyObject* ptr) const {
        Py_DECREF(ptr);
    }
};

template<typename VTRes, typename VTArg>
struct CtypesMapKernel<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    using PyPtr = std::unique_ptr<PyObject, PyDeleter>;

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyPtr pName(PyUnicode_DecodeFSDefault("CtypesMapKernel"));
        std::cout << "Reference count of pName: " << Py_REFCNT(pName.get()) << std::endl;
        PyPtr pModule(PyImport_Import(pName.get()));
        std::cout << "Reference count of pModule: " << Py_REFCNT(pModule.get()) << std::endl;

        if (!pModule) {
            PyErr_Print();
            std::cerr << "Failed to import Python module!" << std::endl;
            return;
        }

        PyPtr pFunc(PyObject_GetAttrString(pModule.get(), "apply_map_function"));
        std::cout << "Reference count of pFunc: " << Py_REFCNT(pFunc.get()) << std::endl;

        if (!PyCallable_Check(pFunc.get())) {
            std::cerr << "Function not callable!" << std::endl;
            return;
        }

        const VTRes* res_data = res->getValues();
        const VTArg* arg_data = arg->getValues();

        if (!res_data || !arg_data) {
            std::cerr << "Null pointer returned from getValues()!" << std::endl;
            return;
        }

        uint64_t data_address_res = reinterpret_cast<uint64_t>(res_data);
        uint64_t data_address_arg = reinterpret_cast<uint64_t>(arg_data);
        
        PyPtr pArgs(Py_BuildValue("KKKKiiss",
                                  address_upper(data_address_res),
                                  address_lower(data_address_res),
                                  address_upper(data_address_arg),
                                  address_lower(data_address_arg),
                                  res->getNumRows(),
                                  res->getNumCols(),
                                  func,
                                  varName));
        
        std::cout << "Reference count of pArgs: " << Py_REFCNT(pArgs.get()) << std::endl;
        
        PyPtr pResult(PyObject_CallObject(pFunc.get(), pArgs.get()));
        std::cout << "Reference count of pResult: " << Py_REFCNT(pResult.get()) << std::endl;

        if (!pResult) {
            PyErr_Print();
        }


    }

    static constexpr uint32_t address_upper(uint64_t data_address) {
        return static_cast<uint32_t>((data_address & 0xFFFFFFFF00000000) >> 32);
    }

    static constexpr uint32_t address_lower(uint64_t data_address) {
        return static_cast<uint32_t>(data_address & 0xFFFFFFFF);
    }
};