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
#include <thread>
#include <sstream>
#include <unistd.h>

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
PyObject* to_python_object(const uint32_t& value)
{
    return PyLong_FromUnsignedLong(value);
}

template <>
PyObject* to_python_object(const uint8_t& value)
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
uint32_t from_python_object<uint32_t>(PyObject* pValue)
{
    return static_cast<uint32_t>(PyLong_AsUnsignedLong(pValue));
}

template <>
uint8_t from_python_object<uint8_t>(PyObject* pValue)
{
    return static_cast<uint8_t>(PyLong_AsUnsignedLong(pValue));
}

template <typename VTRes>
VTRes from_python_object_cutted(PyObject* value);

template <>
double from_python_object_cutted(PyObject* value) {
    double val = PyFloat_AsDouble(value);
    double min_val = -std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::max();
    val = std::max(min_val, std::min(val, max_val));
    return val;
}

template <>
float from_python_object_cutted(PyObject* value) {
    float val = static_cast<float>(PyFloat_AsDouble(value));
    float min_val = -std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::max();
    val = std::max(min_val, std::min(val, max_val));
    return static_cast<double>(val);
}

template <>
int64_t from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromUnsignedLongLong(0x7FFFFFFFFFFFFFFFULL);  // 2^63 - 1
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    int64_t result = PyLong_AsLongLong(modified);
    Py_XDECREF(modified);
    return result;
}

template <>
int32_t from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromUnsignedLong(0x7FFFFFFFUL);  // 2^31 - 1
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    int32_t result = static_cast<int32_t>(PyLong_AsLong(modified));
    Py_XDECREF(modified);
    return result;
}

template <>
int8_t from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromLong(0x7FUL);
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    int8_t result = static_cast<int8_t>(PyLong_AsLong(modified));
    Py_XDECREF(modified);
    return result;
}

template <>
uint64_t from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromUnsignedLongLong(0xFFFFFFFFFFFFFFFFULL);  // 2^64 - 1
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    uint64_t result = PyLong_AsUnsignedLongLong(modified);
    Py_XDECREF(modified);
    return result;
}

template <>
uint8_t from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromUnsignedLong(0xFFUL);  // 2^8 - 1
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    uint8_t result = static_cast<uint8_t>(PyLong_AsUnsignedLong(modified));
    Py_XDECREF(modified);
    return result;
}

template <>
unsigned int from_python_object_cutted(PyObject* value) {
    PyObject *mask = PyLong_FromUnsignedLong(0xFFFFFFFFUL);  // 2^32 - 1
    PyObject *modified = PyNumber_And(value, mask);
    Py_XDECREF(mask);

    unsigned int result = static_cast<unsigned int>(PyLong_AsUnsignedLong(modified));
    Py_XDECREF(modified);
    return result;
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
        } else if (std::is_same<VTArg, uint32_t>::value) {
            return "uint32";
        } else if (std::is_same<VTArg, uint8_t>::value) {
            return "uint8";
        } else {
            throw std::runtime_error("Unsupported data type!");
        }
    }

static std::string generateUniqueID() {
    std::stringstream ss;
    ss << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_MAP_KERNEL_UTILS_H