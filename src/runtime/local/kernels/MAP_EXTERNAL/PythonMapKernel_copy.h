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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_COPY_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_COPY_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <vector>
#include <util/PythonInterpreter.h>
#include <runtime/local/kernels/MAP_EXTERNAL/PythonMapKernelUtils.h>

template<typename DTRes, typename DTArg>
struct PythonMapKernel_copy
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

template<class DTRes, class DTArg>
void pythonMapKernel_copy(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    PythonMapKernel_copy<DTRes,DTArg>::apply(res, arg, func, varName);
}

template<typename VTRes, typename VTArg>
struct PythonMapKernel_copy<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pName = PyUnicode_DecodeFSDefault("PythonMapKernel_copy");
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
            Py_XDECREF(pFunc);
            PyGILState_Release(gstate);
            std::cerr << "Function not callable!" << std::endl;
            return;
        }

        const VTArg* arg_data = arg->getValues();

        if (!arg_data) {
            std::cerr << "Null pointer returned from getValues()!" << std::endl;
            PyGILState_Release(gstate);
            return;
        }

        PyObject* pArgList = PyList_New(arg->getNumRows() * arg->getNumCols());
        for (size_t i = 0; i < arg->getNumRows() * arg->getNumCols(); ++i) {
            PyObject *pValue = to_python_object(arg_data[i]);
            PyList_SetItem(pArgList, i, pValue);
        }

        std::string dtype_arg = get_dtype_name<VTArg>();

        PyObject* pArgs = Py_BuildValue("Oiisss", 
                                        pArgList, 
                                        arg->getNumRows(), 
                                        arg->getNumCols(),
                                        func,
                                        varName,
                                        dtype_arg.c_str()
                                        );
        
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);
        Py_XDECREF(pArgList);

        if (!pResult) {
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        }
        
        int rows = arg->getNumRows();
        int cols = arg->getNumCols();
        std::vector<VTRes> result_data(rows * cols);

        for (int i = 0; i < rows * cols; ++i) {
            PyObject* pValue = PyList_GetItem(pResult, i);
            result_data[i] = from_python_object<VTRes>(pValue);
        }

        Py_XDECREF(pResult);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res->set(i, j, result_data[i * cols + j]);
            }
        }
        PyGILState_Release(gstate);
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_COPY_H