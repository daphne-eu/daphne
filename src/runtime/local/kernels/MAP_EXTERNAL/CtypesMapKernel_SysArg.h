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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SYSARG_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SYSARG_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>
#include <runtime/local/kernels/MAP_EXTERNAL/MapKernelUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct CtypesMapKernel_SysArg {
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypesMapKernel_SysArg(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_SysArg<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct CtypesMapKernel_SysArg<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName) {
    
    PythonInterpreter::getInstance();

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* pName = PyUnicode_FromString("__main__");
    PyObject* pModule = PyImport_Import(pName);
    Py_XDECREF(pName);

    if (!pModule) {
        std::cerr << "Failed to import Python module!" << std::endl;
        PyErr_Print();
        PyGILState_Release(gstate);
        return;
    }

    PyObject *pDict = PyModule_GetDict(pModule);

    std::string functionName;
    if (strstr(func, "def")) { // If the string contains "def"
        // Extract the function name after "def" and before the first parenthesis
        std::string s(func);
        size_t start = s.find("def") + 4; // Skip "def" and whitespace
        size_t end = s.find("(", start);
        functionName = s.substr(start, end - start);

        PyRun_String(func, Py_file_input, pDict, pDict); // Execute the provided code

        if (PyErr_Occurred()) {
            std::cerr << "Error running the provided Python code:" << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
        }
    } else {
        // Treat the provided string as a lambda expression
        std::string lambda_func = "def generated_function(" + std::string(varName) + "):\n\treturn " + func;
        PyRun_String(lambda_func.c_str(), Py_file_input, pDict, pDict);  // Convert lambda to function

        if (PyErr_Occurred()) {
            std::cerr << "Error constructing the lambda function:" << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
        }

        functionName = "generated_function";
    }

    PyObject *pFunc = PyDict_GetItemString(pDict, functionName.c_str());
    if (!PyCallable_Check(pFunc)) {
        std::cerr << "Function '" << functionName << "' not callable or not found!" << std::endl;
        PyGILState_Release(gstate);
        return;
    }

    int numRows = res->getNumRows();
    int numCols = res->getNumCols();
    VTRes* res_data = res->getValues();
    const VTArg* arg_data = arg->getValues();

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            PyObject *pArgs = PyTuple_Pack(1, to_python_object(arg_data[i*numCols + j]));
            PyObject *pResult = PyObject_CallObject(pFunc, pArgs);

            if (!pResult) {
                PyErr_Print();
                PyGILState_Release(gstate);
            } else {
                res_data[i*numCols + j] = from_python_object<VTRes>(pResult);
                Py_XDECREF(pResult);
            }
            
            Py_XDECREF(pArgs);
        }
    }

    Py_XDECREF(pModule);

    PyGILState_Release(gstate);
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SYSARG_H