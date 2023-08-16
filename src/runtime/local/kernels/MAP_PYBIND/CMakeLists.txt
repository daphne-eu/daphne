set (CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.21)
project(mapExternalPL_PyBindKernel)
message(STATUS "PyBind CMake Build")

# Find the required libraries for your Pybind11 module (Pybind11 and Python)
find_package(pybind11 REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

# Find and link against the SPDLOG library
find_package(spdlog CONFIG REQUIRED)

# Add the Pybind11 module
pybind11_add_module(mapExternalPL_PyBindKernel MODULE PyBindMapKernel.h)
#pybind11_add_module(mapExternalPL_PyBindKernel MODULE PyBindMapKernel_Copy.h)

# Link the Pybind11 module with the existing libraries (e.g., AllKernels)
#find_library(ALL_KERNELS_LIB NAMES AllKernels PATHS "../../../../../lib")
target_link_libraries(mapExternalPL_PyBindKernel PUBLIC ${ALL_KERNELS_LIB} spdlog::spdlog)
#target_link_libraries(mapExternalPL_PyBindKernel_Copy PUBLIC ${ALL_KERNELS_LIB} spdlog::spdlog)

# Set the output directory for the shared library
set_target_properties(mapExternalPL_PyBindKernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../../../../lib)
#set_target_properties(mapExternalPL_PyBindKernel_Copy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../../../../lib)