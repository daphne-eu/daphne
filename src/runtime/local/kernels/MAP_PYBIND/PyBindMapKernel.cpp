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

#include <runtime/local/kernels/MAP_PYBIND/PyBindMapKernel.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

static void PyBindMapKernel::applyMapFunction(DenseMatrix<VTRes>*& res, const DenseMatrix<VTArg>* arg, void* func) {
    // Get the raw pointers to the data of input and output DenseMatrices
    const VTArg* arg_data = arg->getValuesSharedPtr().get();
    VTRes* res_data = res->getValuesSharedPtr().get();
    std::cout << "arg_data pointer type before: " << typeid(arg_data).name() << std::endl;
    std::cout << "res_data pointer type before: " << typeid(res_data).name() << std::endl;

    // Get the number of rows and columns in the input and output DenseMatrices
    size_t rows = arg->getNumRows();
    size_t cols = arg->getNumCols();

    // Import the Python Map function from map_function.py
    py::object map_function = py::module::import("map_function").attr("map_kernel_pybind");

    // Create numpy arrays from the raw pointers without making copies
    py::array_t<VTArg> py_input(rows * cols, arg_data);
    py_input.resize({rows, cols});

    py::array_t<VTRes> py_output(rows * cols, res_data);
    py_output.resize({rows, cols});

    std::cout << "py_input dtype: " << py_input.attr("dtype").attr("name") << std::endl;
    std::cout << "py_output dtype: " << py_output.attr("dtype").attr("name") << std::endl;

    // Convert the function pointer into a String Representation
    std::string func_str = reinterpret_cast<char*>(func);
    
    // Call the Python Map function on the input matrix and function pointer
    try
    {
        py::object map_function = py::module::import("map_function").attr("map_kernel_pybind");
        map_function(py_input, py_output, func_str);
        std::cout << "arg_data pointer type after: " << typeid(arg_data).name() << std::endl;
        std::cout << "res_data pointer type after: " << typeid(res_data).name() << std::endl;
    }
    catch(const py::error_already_set& ex)
    {
        PyErr_Print();
    }    
}