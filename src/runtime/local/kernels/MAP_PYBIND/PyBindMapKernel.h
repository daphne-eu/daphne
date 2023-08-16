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
struct PyBindMapKernel
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void pyBindMapKernel(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    PyBindMapKernel<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct PyBindMapKernel<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
       // Get the raw pointers to the data of input and output DenseMatrices
        const VTArg* arg_data = arg->getValuesSharedPtr().get();
        VTRes* res_data = res->getValuesSharedPtr().get();
        std::cout << "arg_data: " << arg_data << ", res_data: " << res_data << std::endl;

        // Get the number of rows and columns in the input and output DenseMatrices
        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

        // Create numpy arrays from the raw pointers without making copies
        try {
            py::array_t<VTArg> py_input(rows * cols, arg_data);
            py_input.resize({rows, cols});

            py::array_t<VTRes> py_output(rows * cols, res_data);
            py_output.resize({rows, cols});

            std::cout << "py_input rows: " << py_input.shape(0) << ", cols: " << py_input.shape(1) << std::endl;
            std::cout << "py_output rows: " << py_output.shape(0) << ", cols: " << py_output.shape(1) << std::endl;

            // Import the Python Map function from map_function.py
            py::object map_function = py::module::import("map_function").attr("map_kernel_pybind");
    
            // Call the Python Map function on the input matrix and function pointer
            map_function(py_input, py_output, func, varName);
            std::cout << "Python function called successfully" << std::endl;
        } 
        catch (const std::exception& ex)
        {
            std::cerr << "An exception occurred while creating or resizing py_input: " << ex.what() << std::endl;
        }
    }
};