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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_CSV_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_CSV_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/MAP_EXTERNAL/PythonMapKernelUtils.h>
#include <Python.h>
#include <fstream>
#include <util/PythonInterpreter.h>
#include <cstdio>
#include <sstream>
#include <string>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct PythonMapKernel_csv
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void pythonMapKernel_csv(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    PythonMapKernel_csv<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct PythonMapKernel_csv<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pName = PyUnicode_DecodeFSDefault("PythonMapKernel_csv");
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

        std::string id = generateUniqueID();
        const std::string inputFile = "input_data_" + id + ".csv";
        const std::string outputFile = "output_data" + id + ".csv";

        std::ofstream ofs(inputFile);
        for (size_t i = 0; i < arg->getNumRows(); ++i) {
            for (size_t j = 0; j < arg->getNumCols(); ++j) {
                std::string valueStr = std::to_string(arg->get(i, j));
        
                ofs << valueStr << (j == arg->getNumCols() - 1 ? "\n" : ",");
            }
        }

        ofs.close();

        std::string dtype = get_dtype_name<VTArg>();

        PyObject* pArgs = Py_BuildValue("ssiisss",
                                        inputFile.c_str(),
                                        outputFile.c_str(),
                                        res->getNumRows(),
                                        res->getNumCols(),
                                        func,
                                        varName,
                                        dtype.c_str());

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        } else {
            PyObject* pStatus = PyTuple_GetItem(pResult, 0);
            PyObject* pMessage = PyTuple_GetItem(pResult, 1);
            bool success = PyObject_IsTrue(pStatus);

            if (!success) {
                std::cerr << PyUnicode_AsUTF8(pMessage) << std::endl;
                Py_XDECREF(pResult);
                if (std::remove(inputFile.c_str()) != 0) {
                    perror("Error deleting input csv file");
                }
                PyGILState_Release(gstate);
                return;
            }

            Py_XDECREF(pResult);
        }

        std::ifstream ifs(outputFile);
        for (size_t i = 0; i < res->getNumRows(); ++i) {
            for (size_t j = 0; j < res->getNumCols(); ++j) {
                VTRes value = readValue(ifs);
        
                // If not on the last column, skip the comma
                if (j < res->getNumCols() - 1) {
                    ifs.ignore(); // Ignore the comma (',')
                }

                res->set(i, j, value);
            }
            // This will consume the newline at the end of each row.
            ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        ifs.close();
        cleanupFiles(inputFile, outputFile);
        PyGILState_Release(gstate);
    }

    static VTRes readValue(std::ifstream& ifs)
    {
        VTRes value;
        if (std::is_same<VTRes, uint8_t>::value || std::is_same<VTRes, int8_t>::value) {
            int temp; // Use a temp int to store the parsed value
            ifs >> temp;
            value = static_cast<VTRes>(temp); // Cast back to uint8_t
        } else {
            ifs >> value;
        }
        return value;
    }

    static void cleanupFiles(const std::string& inputFile, const std::string& outputFile) {
        if (std::remove(inputFile.c_str()) != 0) {
            perror("Error deleting temp input csv file");
        }
        if (std::remove(outputFile.c_str()) != 0) {
            perror("Error deleting temp output csv file");
        }
    }

};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_CSV_H