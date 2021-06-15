find_package(Threads)

set(SOURCES ProtoDataConverter.cpp)
set(LIBS DataStructures Proto)

add_library(ProtoDataConverter ${SOURCES})
target_link_libraries(ProtoDataConverter PRIVATE ${LIBS})

# FIXME: use DaphneIrExecutor target instead
set(SOURCES WorkerImpl.cpp ../../../compiler/execution/DaphneIrExecutor.cpp)

#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRDaphne
        MLIRDaphneTransforms
        MLIRExecutionEngine
        #DaphneIrExecutor
        DataStructures
        ProtoDataConverter
        Proto
        )

# TODO: check why SHARED is required
add_library(WorkerImpl SHARED ${SOURCES})
#llvm_update_compile_flags(WorkerImpl)
target_link_libraries(WorkerImpl PRIVATE ${LIBS})
#mlir_check_all_link_libraries(WorkerImpl)

add_executable(DistributedWorker main.cpp)
target_link_libraries(DistributedWorker
        PRIVATE
        Proto
        WorkerImpl
        MLIRDaphne)