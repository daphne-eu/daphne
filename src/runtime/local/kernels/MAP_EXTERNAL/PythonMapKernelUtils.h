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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_MAP_KERNEL_UTILS_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_MAP_KERNEL_UTILS_H
#include <Python.h>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>

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
    return PyFloat_FromDouble(static_cast<double>(value));
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

template <typename VTArg>
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

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_MAP_KERNEL_UTILS_H