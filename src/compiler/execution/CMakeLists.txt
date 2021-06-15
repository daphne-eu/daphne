set(SOURCES DaphneIrExecutor.cpp DaphneIrExecutor.h)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRDaphne
        MLIRDaphneTransforms
        MLIRExecutionEngine
        )

# TODO: check why SHARED is required
add_llvm_library(DaphneIrExecutor SHARED ${SOURCES}
        DEPENDS
        MLIRDaphneOpsIncGen
        # The kernels are linked at runtime.
        AllKernels
        )

llvm_update_compile_flags(DaphneIrExecutor)
target_link_libraries(DaphneIrExecutor PUBLIC ${LIBS})
mlir_check_all_link_libraries(DaphneIrExecutor)
