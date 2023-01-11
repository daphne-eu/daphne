get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}

        MLIRDaphne
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        # MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        # MLIRMemRefDialect
        # MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        )
add_llvm_executable(daphne-opt daphne-opt.cpp)

llvm_update_compile_flags(daphne-opt)
target_link_libraries(daphne-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(daphne-opt)
