#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>
#include <fstream>
#include <cstdint>

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

        const std::string inputFile = "input_data.bin";
        const std::string outputFile = "output_data.bin";

        // Serialize data to a binary file
        std::ofstream output(inputFile, std::ios::binary);
        output.write(reinterpret_cast<const char *>(arg->getValues()), arg->getNumRows() * arg->getNumCols() * sizeof(VTArg));
        output.close();

        // Call Python function to process the data
        PythonInterpreter::getInstance();
        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_BinaryData");
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

        PyObject* pArgs = Py_BuildValue("sssiiss", inputFile.c_str(), outputFile.c_str(), res->getNumRows(), res->getNumCols(), func, varName);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);
        if (!pResult) {
            PyErr_Print();
        } else {
            Py_XDECREF(pResult);
        }

        // Deserialize result from binary file
        std::ifstream input(outputFile, std::ios::binary);
        input.read((char *)res->getValues(), res->getNumRows() * res->getNumCols() * sizeof(VTRes));
        input.close();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_BINARYDATA_BINARYDATAMAPKERNEL_H