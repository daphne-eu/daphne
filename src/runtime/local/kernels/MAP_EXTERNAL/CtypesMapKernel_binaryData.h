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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>
#include <fstream>
#include <cstdint>
#include <cstdio>

template<typename DTRes, typename DTArg>
struct CtypesMapKernel_binaryData
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

template<class DTRes, class DTArg>
void ctypesMapKernel_binaryData(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_binaryData<DTRes,DTArg>::apply(res, arg, func, varName);
}

template<typename VTRes, typename VTArg>
struct CtypesMapKernel_binaryData<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();
        
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        const std::string inputFile = "input_data.bin";
        const std::string outputFile = "output_data.bin";

        std::ofstream output(inputFile, std::ios::binary);
        output.write(reinterpret_cast<const char *>(arg->getValues()), arg->getNumRows() * arg->getNumCols() * sizeof(VTArg));
        output.close();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_binaryData");
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

        std::string dtype = get_dtype_name();

        PyObject* pArgs = Py_BuildValue("ssiisss", 
                                        inputFile.c_str(), 
                                        outputFile.c_str(), 
                                        res->getNumRows(), 
                                        res->getNumCols(), 
                                        func, 
                                        varName, 
                                        dtype.c_str()
                                        );
        
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);
        if (!pResult) {
            PyErr_Print();
            PyGILState_Release(gstate);
        } else {
            Py_XDECREF(pResult);
        }

        std::ifstream input(outputFile, std::ios::binary);
        input.read((char *)res->getValues(), res->getNumRows() * res->getNumCols() * sizeof(VTRes));
        input.close();

        if (std::remove(inputFile.c_str()) != 0) {
            perror("Error deleting input file");
        }
        if (std::remove(outputFile.c_str()) != 0) {
            perror("Error deleting output file");
        }

        PyGILState_Release(gstate);
    }

    static std::string get_dtype_name() {
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
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H