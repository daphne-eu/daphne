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

// ****************************************************************************
// PythonInterpreter Singleton
// ****************************************************************************
struct PythonInterpreter {
public:
    static PythonInterpreter& getInstance() {
        static PythonInterpreter instance;
        return instance;
    }
private:
    PythonInterpreter() {
        Py_Initialize();
        std::string kernelPath = std::string(PROJECT_SOURCE_DIR) + "/src/runtime/local/kernels/MAP_CTYPES";
        addPath(kernelPath.c_str());
    }
    ~PythonInterpreter() {
        Py_Finalize();
    }

    // Add paths to sys.path
    void addPath(const char* newPath) {
        PyObject* pSysPath = PySys_GetObject("path");
        PyObject* pPath = PyUnicode_FromString(newPath);
        if (pSysPath && pPath && PyList_Append(pSysPath, pPath) == 0) {
            Py_DECREF(pPath);
        } else {
            throw std::runtime_error("Failed to add path to sys.path");
        }
    }

    // Non-copyable and non-moveable
    PythonInterpreter(const PythonInterpreter&) = delete;
    PythonInterpreter& operator=(const PythonInterpreter&) = delete;
    PythonInterpreter(PythonInterpreter&&) = delete;
    PythonInterpreter& operator=(PythonInterpreter&&) = delete;
};
#endif // PYTHON_INTERPRETER_H