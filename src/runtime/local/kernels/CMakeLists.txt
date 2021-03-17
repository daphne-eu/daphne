add_library(PrintKernels SHARED PrintKernels.cpp)
target_link_libraries(PrintKernels)

add_library(LinAlgKernels SHARED LinAlgKernels.cpp)
target_link_libraries(LinAlgKernels)