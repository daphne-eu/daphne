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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#pragma once

#include <runtime/local/kernels/MAP_NUMPY/NumpyTypeString.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct NumpyMapKernel
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char * varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void numpyMapKernel(DTRes *& res, const DTArg * arg, const char* func, const char * varName) {
    NumpyMapKernel<DTRes,DTArg>::apply(res, arg, func, varName);
}

void initialize_numpy() {
    import_array1();
}


// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template <typename VTRes, typename VTArg>
struct NumpyMapKernel<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char * func, const char * varName)
    {
        // Initialize the Python Interpreter
        Py_Initialize();

        // Import numpy
        initialize_numpy();

        // Get the map_function from your Python script
        PyObject* pName = PyUnicode_DecodeFSDefault("NumpyMapKernel");
        PyObject* pModule = PyImport_Import(pName);
        Py_DECREF(pName);
        PyObject* pFunc = PyObject_GetAttrString(pModule, "map_function");

        // Call the map_function with the appropriate arguments
        PyObject* pArgs = PyTuple_Pack(7,
                                        PyLong_FromVoidPtr(arg->getValuesSharedPtr().get()),
                                        PyLong_FromVoidPtr(res->getValuesSharedPtr().get()),
                                        Py_BuildValue("(ii)", arg->getNumRows(), arg->getNumCols()),
                                        Py_BuildValue("(ii)", res->getNumRows(), res->getNumCols()),
                                        PyUnicode_FromString(func),
                                        PyUnicode_FromString(varName),
                                        PyUnicode_FromString(NumpyTypeString<VTArg>::value));

        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pValue);

        // Clean up
        Py_DECREF(pFunc);
        Py_DECREF(pModule);

        // Finalize the Python Interpreter
        Py_Finalize();
    }
};