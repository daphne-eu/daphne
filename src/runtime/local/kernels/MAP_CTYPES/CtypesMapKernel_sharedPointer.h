#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_SHAREDPOINTER_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_SHAREDPOINTER_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct CtypesMapKernel_SharedPointer
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypesMapKernel_SharedPointer(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_SharedPointer<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct CtypesMapKernel_SharedPointer<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_SharedPointer");
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

        auto res_sp = res->getValuesSharedPtr();
        auto arg_sp = arg->getValuesSharedPtr();

        std::string orig_dtype_arg = get_dtype_name();
        std::string orig_dtype_res = orig_dtype_arg; // Assuming VTArg and VTRes have the same data type

        PyObject* pArgs = Py_BuildValue("LLiissss",
                                        reinterpret_cast<int64_t>(res_sp.get()),
                                        reinterpret_cast<int64_t>(arg_sp.get()),
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
        } else {
            Py_XDECREF(pResult);
        }
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

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_CTYPES_CTYPESMAPKERNEL_SHAREDPOINTER_H