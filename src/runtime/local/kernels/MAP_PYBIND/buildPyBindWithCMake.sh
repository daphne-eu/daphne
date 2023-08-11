# Run the Python command and capture its output into the variable
PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Now, run the CMake command using the captured variable
cmake -S .. -B . -DCMAKE_PREFIX_PATH=$PYBIND11_CMAKE_DIR

make