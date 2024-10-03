#!/usr/bin/env bash

export ROOTDIR=$PWD
export NUM_VCORES=$(nproc)
JDK_VERSION=11.0.22
JDK_BUILD=7
export JAVA_HOME=$ROOTDIR/jdk-$JDK_VERSION+$JDK_BUILD
# export PATH=$JAVA_HOME/bin:${ROOTDIR}/installed/bin:$PATH
export DLC=$ROOTDIR/DLC

export CMAKE_VERSION=3.29.0
export GCC_VERSION=13.2.0
export NINJA_VERSION=1.11.1
export RE2C_VERSION=3.1
export PYTHON_VERSION=3.12.2
export LLVM_VERSION=18.1.2
export BINUTILS_VERSION=2.42
export CUDA12_VERSION=12.4
export CUDA_VERSION=$CUDA12_VERSION
export CUDA12_PACKAGE=cuda_12.4.0_550.54.14_linux.run
export CUDNN9_PACKAGE=cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
export CUDA_PATH=$ROOTDIR/installed/cuda-$CUDA_VERSION

export PATH=$CUDA_PATH/bin:$PWD/installed/bin:$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$PWD/installed/lib:$PWD/installed/lib64:$LD_LIBRARY_PATH
export LD_RUN_PATH=$LD_LIBRARY_PATH


# rhel für gcc compile:
# install xz, libmpc-devel gmp-devel mpfr-devel

# exit
# run docker:
#!/usr/bin/env bash
# docker run --rm -it -v $PWD:/workdir --entrypoint=/workdir/commands.sh daphne-rhel bash
