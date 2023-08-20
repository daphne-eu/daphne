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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_COPY_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_COPY_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <vector>
#include <util/PythonInterpreter.h>

template<typename DTRes, typename DTArg>
struct CtypesMapKernel_copy
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

template<class DTRes, class DTArg>
void ctypesMapKernel_copy(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_copy<DTRes,DTArg>::apply(res, arg, func, varName);
}

template<typename VTRes, typename VTArg>
struct CtypesMapKernel_copy<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_copy");
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

        const VTArg* arg_data = arg->getValues();

        if (!arg_data) {
            std::cerr << "Null pointer returned from getValues()!" << std::endl;
            return;
        }

        PyObject* pArgList = PyList_New(arg->getNumRows() * arg->getNumCols());
        for (size_t i = 0; i < arg->getNumRows() * arg->getNumCols(); ++i) {
            PyObject* pValue = PyFloat_FromDouble(arg_data[i]);
            PyList_SetItem(pArgList, i, pValue);
        }

        PyObject* pArgs = Py_BuildValue("Oii", pArgList, arg->getNumRows(), arg->getNumCols());
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);
        Py_XDECREF(pArgList);

        if (!pResult) {
            PyErr_Print();
        } else {
            int rows = arg->getNumRows();
            int cols = arg->getNumCols();
            std::vector<VTRes> result_data(rows * cols);

            for (int i = 0; i < rows * cols; ++i) {
                PyObject* pValue = PyList_GetItem(pResult, i);
                result_data[i] = static_cast<VTRes>(PyFloat_AsDouble(pValue));
                Py_DECREF(pValue);
        }

            Py_XDECREF(pResult);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    res->set(i, j, result_data[i * cols + j]);
                }
            }
        }

    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_COPY_H