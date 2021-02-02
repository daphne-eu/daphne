# *****************************************************************************
# antlr4-related
# *****************************************************************************

CMAKE_MINIMUM_REQUIRED(VERSION 3.7 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)

# required if linking to static library
add_definitions(-DANTLR4CPP_STATIC)

# set variable pointing to the antlr tool that supports C++
# this is not required if the thirdparty file can be found under PATH environment
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/thirdparty/antlr-4.9.1-complete.jar)

#set(ANTLR4_WITH_STATIC_CRT OFF)
# add external build for antlrcpp
include(ExternalAntlr4Cpp)
# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

# add macros to generate ANTLR Cpp code from grammar
find_package(ANTLR REQUIRED)

# Call macro to add lexer and grammar to your build dependencies.
antlr_target(DaphneGrammarLexer DaphneLexer.g4 LEXER
        PACKAGE daphne_antlr)
antlr_target(DaphneGrammarParser DaphneParser.g4 PARSER
        PACKAGE daphne_antlr
        DEPENDS_ANTLR DaphneGrammarLexer
        COMPILE_FLAGS -lib ${ANTLR_DaphneGrammarLexer_OUTPUT_DIR}
        VISITOR)

# include generated files in project environment
include_directories(${ANTLR_DaphneGrammarLexer_OUTPUT_DIR})
include_directories(${ANTLR_DaphneGrammarParser_OUTPUT_DIR})

# *****************************************************************************
# daphnec
# *****************************************************************************

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs} # contains MLIRDaphne
        ${conversion_libs} # contains MLIRDaphneTransforms
        MLIRExecutionEngine
        antlr4_static
)

# enable exceptions handling
set(LLVM_ENABLE_EH ON)

add_llvm_executable(daphnec
        daphnec.cpp
        MLIRGenVisitors.cpp
        Builtins.cpp
        ${ANTLR_DaphneGrammarLexer_CXX_OUTPUTS}
        ${ANTLR_DaphneGrammarParser_CXX_OUTPUTS}

        DEPENDS
        MLIRDaphneOpsIncGen
        # The kernels are linked at runtime.
        PrintKernels
        LinAlgKernels
)

llvm_update_compile_flags(daphnec)
target_link_libraries(daphnec PRIVATE ${LIBS})

mlir_check_all_link_libraries(daphnec)