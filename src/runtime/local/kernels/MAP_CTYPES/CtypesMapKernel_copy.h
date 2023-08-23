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


 //Helper Method for extracting from Python Objects
template <typename VTArg>
PyObject* to_python_object(const VTArg& value);

template <>
PyObject* to_python_object(const double& value)
{
    return PyFloat_FromDouble(value);
}

template <>
PyObject* to_python_object(const float& value)
{
    return PyFloat_FromDouble(value);
}

template <>
PyObject* to_python_object(const int64_t& value)
{
    return PyLong_FromLongLong(value);
}

template <>
PyObject* to_python_object(const int32_t& value)
{
    return PyLong_FromLong(value);
}

template <>
PyObject* to_python_object(const int8_t& value)
{
    return PyLong_FromLong(value);
}

template <>
PyObject* to_python_object(const uint64_t& value)
{
    return PyLong_FromUnsignedLongLong(value);
}

template <>
PyObject* to_python_object(const uint8_t& value)
{
    return PyLong_FromUnsignedLong(value);
}

template <>
PyObject* to_python_object(const unsigned int& value)
{
    return PyLong_FromUnsignedLong(value);
}

//Helper Methods to transfer to Python Objects
template <typename VTRes>
VTRes from_python_object(PyObject* pValue);

template <>
double from_python_object<double>(PyObject* pValue)
{
    return PyFloat_AsDouble(pValue);
}

template <>
float from_python_object<float>(PyObject* pValue)
{
    return static_cast<float>(PyFloat_AsDouble(pValue));
}

template <>
int64_t from_python_object<int64_t>(PyObject* pValue)
{
    return PyLong_AsLongLong(pValue);
}

template <>
int32_t from_python_object<int32_t>(PyObject* pValue)
{
    return static_cast<int32_t>(PyLong_AsLong(pValue));
}

template <>
int8_t from_python_object<int8_t>(PyObject* pValue)
{
    return static_cast<int8_t>(PyLong_AsLong(pValue));
}

template <>
uint64_t from_python_object<uint64_t>(PyObject* pValue)
{
       return PyLong_AsUnsignedLongLong(pValue);
}

template <>
uint8_t from_python_object<uint8_t>(PyObject* pValue)
{
    return static_cast<uint8_t>(PyLong_AsUnsignedLong(pValue));
}

template <>
unsigned int from_python_object<unsigned int>(PyObject* pValue)
{
    return static_cast<unsigned int>(PyLong_AsUnsignedLong(pValue));
}

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
            PyObject* pValue = to_python_object<VTArg>(arg_data[i]);
            PyList_SetItem(pArgList, i, pValue);
        }

        std::string dtype_arg = get_dtype_name();

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
        } else {
            int rows = arg->getNumRows();
            int cols = arg->getNumCols();
            std::vector<VTRes> result_data(rows * cols);

            for (int i = 0; i < rows * cols; ++i) {
                PyObject* pValue = PyList_GetItem(pResult, i);
                result_data[i] = from_python_object<VTRes>(pValue);
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

    //Helper Method for defining, weather it is a String or not
    static std::string get_dtype_name() 
    {
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

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_COPY_H