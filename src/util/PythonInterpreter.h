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

#ifndef PYTHON_INTERPRETER_H
#define PYTHON_INTERPRETER_H
#include <Python.h>
#include <stdexcept>
#include <iostream>
#include <stack>

// ****************************************************************************
// PythonInterpreter Singleton
// ****************************************************************************
struct PythonInterpreter {
public:

    static PythonInterpreter& getInstance() {
        static PythonInterpreter pythonInterpreter;
        return pythonInterpreter;
    }

    static void finalizeInterpreter() {
        getInstance().finalize();
    }

    static void initializeInterpreter() {
        getInstance().init();
    }

    static void addPath(const char* newPath) {
        PyObject* pSysPath = PySys_GetObject("path");
        if (!pSysPath) {
            throw std::runtime_error("Failed to get sys.path");
        }
        PyObject* pPath = PyUnicode_FromString(newPath);
        if (!pPath) {
            throw std::runtime_error("Failed to create Python string from path");
        }

        // Add the path to sys.path
        if (PyList_Append(pSysPath, pPath) != 0) {
            Py_XDECREF(pPath);
            throw std::runtime_error("Failed to add path to sys.path");
        }

        Py_XDECREF(pPath);
    }

private:

    PythonInterpreter() {
        init();
    }
    
    ~PythonInterpreter() {
        finalize();
    }

    static void init() {
        if (!Py_IsInitialized()) {
            Py_Initialize();

            std::string kernelPath_map_ctypes = std::string(PROJECT_SOURCE_DIR) + "/src/runtime/local/kernels/MAP_CTYPES";          
            addPath(kernelPath_map_ctypes.c_str());

            std::string kernelPath_map_numpy = std::string(PROJECT_SOURCE_DIR) + "/src/runtime/local/kernels/MAP_NUMPY";          
            addPath(kernelPath_map_numpy.c_str());
        }
    }

    static void finalize() {
        if (Py_IsInitialized()) {
            Py_Finalize();
        }
    }

    // Non-copyable and non-moveable
    PythonInterpreter(const PythonInterpreter&) = delete;
    PythonInterpreter& operator=(const PythonInterpreter&) = delete;
    PythonInterpreter(PythonInterpreter&&) = delete;
    PythonInterpreter& operator=(PythonInterpreter&&) = delete;
};

#endif // PYTHON_INTERPRETER_H