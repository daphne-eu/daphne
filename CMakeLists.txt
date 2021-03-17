cmake_minimum_required(VERSION 3.13.4)
project(daphne-prototype LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

# TODO Don't hardcode these paths. See "src/parser/daphnedsl/CMakeLists.txt".
include_directories(${PROJECT_BINARY_DIR}/src/parser/daphnedsl/antlr4_runtime/src/antlr4_runtime/runtime/Cpp/runtime/src)
include_directories(${PROJECT_BINARY_DIR}/src/parser/daphnedsl/antlr4cpp_generated_src/DaphneLexer)
include_directories(${PROJECT_BINARY_DIR}/src/parser/daphnedsl/antlr4cpp_generated_src/DaphneParser)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src/api/cli)
add_subdirectory(src/compiler/lowering)
add_subdirectory(src/ir/daphneir)
add_subdirectory(src/parser/daphnedsl)
add_subdirectory(src/runtime/local/kernels)