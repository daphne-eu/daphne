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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_SHAREDMEM_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_SHAREDMEM_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/MAP_EXTERNAL/PythonMapKernelUtils.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct PythonMapKernel_Shared_Mem
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void pythonMapKernel_Shared_Mem(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    PythonMapKernel_Shared_Mem<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct PythonMapKernel_Shared_Mem<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pName = PyUnicode_DecodeFSDefault("PythonMapKernel_shared_mem");
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

        const VTRes* res_data = res->getValues();
        const VTArg* arg_data = arg->getValues();

        if (!res_data || !arg_data) {
            std::cerr << "Null pointer returned from getValues()!" << std::endl;
            return;
        }

        uint64_t data_address_res = reinterpret_cast<uint64_t>(res_data);
        uint64_t data_address_arg = reinterpret_cast<uint64_t>(arg_data);

        std::string orig_dtype_arg = get_dtype_name<VTArg>();
        std::string orig_dtype_res = orig_dtype_arg; // Assuming VTArg and VTRes have the same data type

        PyObject* pArgs = Py_BuildValue("KKKKiissss",
                                        address_upper(data_address_res),
                                        address_lower(data_address_res),
                                        address_upper(data_address_arg),
                                        address_lower(data_address_arg),
                                        res->getNumRows(),
                                        res->getNumCols(),
                                        func,
                                        varName,
                                        orig_dtype_arg.c_str(),
                                        orig_dtype_res.c_str()
                                        );

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        } else {
            Py_XDECREF(pResult);
        }

        PyGILState_Release(gstate);
    }

    static constexpr uint32_t address_upper(uint64_t data_address) {
        return static_cast<uint32_t>((data_address & 0xFFFFFFFF00000000) >> 32);
    }

    static constexpr uint32_t address_lower(uint64_t data_address) {
        return static_cast<uint32_t>(data_address & 0xFFFFFFFF);
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_PYTHONMAPKERNEL_SHAREDMEM_H