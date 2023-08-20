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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CSVMAPKERNEL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CSVMAPKERNEL_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <fstream>
#include <util/PythonInterpreter.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct CtypesMapKernel_csv
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypesMapKernel_csv(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_csv<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct CtypesMapKernel_csv<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_csv");
        PyObject* pModule = PyImport_Import(pName);
        Py_XDECREF(pName);

        if (!pModule) {
            std::cerr << "Failed to import Python module!" << std::endl;
            PyErr_Print();
            return;
        }

        PyObject* pFunc = PyObject_GetAttrString(pModule, "apply_map_function");
        Py_XDECREF(pModule);

        if (!PyCallable_Check(pFunc)) {
            Py_XDECREF(pFunc);
            std::cerr << "Function not callable!" << std::endl;
            return;
        }

        std::ofstream ofs("input.csv");
        for (size_t i = 0; i < arg->getNumRows(); ++i) {
            for (size_t j = 0; j < arg->getNumCols(); ++j) {
                ofs << arg->get(i, j) << (j == arg->getNumCols() - 1 ? "\n" : ",");
            }
        }
        ofs.close();

        PyObject* pArgs = Py_BuildValue("siiss",
                                        "input.csv",
                                        res->getNumRows(),
                                        res->getNumCols(),
                                        func,
                                        varName);

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
        } else {
            Py_XDECREF(pResult);
        }

        std::ifstream ifs("output.csv");
        for (size_t i = 0; i < res->getNumRows(); ++i) {
            for (size_t j = 0; j < res->getNumCols(); ++j) {
                double value;
                ifs >> value;
                res->set(i, j, value);
            }
        }
        ifs.close();
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CSVMAPKERNEL_H