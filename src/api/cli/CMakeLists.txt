get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRDaphne
        MLIRDaphneTransforms
        MLIRExecutionEngine
        DaphneDSLParser
)

# enable exceptions handling
set(LLVM_ENABLE_EH ON)

add_llvm_executable(daphnec
        daphnec.cpp

        DEPENDS
        MLIRDaphneOpsIncGen
        # The kernels are linked at runtime.
        PrintKernels
        LinAlgKernels
)

llvm_update_compile_flags(daphnec)
target_link_libraries(daphnec PRIVATE ${LIBS})

mlir_check_all_link_libraries(daphnec)