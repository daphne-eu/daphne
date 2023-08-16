#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <runtime/local/datastructures/DenseMatrix.h>
namespace py = pybind11;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct PyBindMapKernel_Copy
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void pyBindMapKernel_Copy(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    PyBindMapKernel_Copy<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct PyBindMapKernel_Copy<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        
        try {
            py::array_t<VTArg> py_input({rows, cols});
            py::array_t<VTRes> py_output({rows, cols});

            auto py_input_ptr = py_input.mutable_data();
            auto py_output_ptr = py_output.mutable_data();

            const VTArg* arg_data = arg->getValuesSharedPtr().get();
            std::memcpy(py_input_ptr, arg_data, sizeof(VTArg) * rows * cols);

            VTRes* res_data = res->getValuesSharedPtr().get();
            std::memcpy(py_output_ptr, res_data, sizeof(VTRes) * rows * cols);

            py::object map_function = py::module::import("map_function").attr("map_kernel_pybind_copy");
    
            py::array_t<VTRes> py_modified_output = map_function(py_input, py_output, func, varName).cast<py::array_t<VTRes>>();

            VTRes* modified_output_data = py_modified_output.mutable_data();
            std::memcpy(res_data, modified_output_data, sizeof(VTRes) * rows * cols);

            std::cout << "Python function called successfully" << std::endl;
        } 
        catch (const std::exception& ex)
        {
            std::cerr << "An exception occurred: " << ex.what() << std::endl;
        }
    }
};