#!/usr/bin/env bash

# Stop immediately if any command fails.
set -e

source env.sh

# download cache:
export DLC=$ROOTDIR/DLC
if [ !  -d $DLC ]; then
    echo Creating DLC dir
    mkdir -p $DLC
fi

mkdir -p $ROOTDIR/installed

export GMP_VERSION=6.3.0
export MPFR_VERSION=4.2.1
export MPC_VERSION=1.3.1
export ELF_VERSION=0.191
export BZIP2_VERSION=1.0.8

if [ ! -f $DLC/gmp-${GMP_VERSION}.tar.xz ];then
    wget https://gmplib.org/download/gmp/gmp-${GMP_VERSION}.tar.xz -P $DLC
    tar xf $DLC/gmp-${GMP_VERSION}.tar.xz
    cd gmp-${GMP_VERSION}
    ./configure --enable-shared --enable-static --prefix=$ROOTDIR/installed
    make -j$NUM_VCORES
    make check
    make install
    cd -
fi

if [ ! -f $DLC/mpfr-${MPFR_VERSION}.tar.xz ];then
    wget https://www.mpfr.org/mpfr-current/mpfr-${MPFR_VERSION}.tar.xz -P $DLC
    tar xf $DLC/mpfr-${MPFR_VERSION}.tar.xz
    cd mpfr-${MPFR_VERSION}
    ./configure --enable-shared --enable-static --prefix=$ROOTDIR/installed --with-gmp=$ROOTDIR/installed
    make -j$NUM_VCORES
    make install
    cd -
fi

if [ ! -f $DLC/mpc-${MPC_VERSION}.tar.gz ];then
    wget https://ftp.gnu.org/gnu/mpc/mpc-${MPC_VERSION}.tar.gz -P $DLC
    tar xf $DLC/mpc-${MPC_VERSION}.tar.gz
    cd mpc-${MPC_VERSION}
    ./configure --enable-shared --enable-static --prefix=$ROOTDIR/installed --with-gmp=$ROOTDIR/installed --with-mpfr=$ROOTDIR/installed
    make -j$NUM_VCORES
    make install 
    cd -
fi


#bzip needs this change in the Makefile (otherwise the python compile fails)
#CFLAGS=-Wall -Winline -O2 -g $(BIGFILES)
#CFLAGS= -O3 -fomit-frame-pointer -funroll-loops -fPIC

# if [ ! -f $DLC/bzip2-${BZIP2_VERSION}.tar.gz ];then
#     wget https://sourceware.org/pub/bzip2/bzip2-${BZIP2_VERSION}.tar.gz -P $DLC
#     tar xf $DLC/bzip2-${BZIP2_VERSION}.tar.gz
#     cd bzip2-${BZIP2_VERSION}
#     make -j$NUM_VCORES
#     make install PREFIX=$ROOTDIR/installed
#     cd -
# fi

#optional stuff for libelf
# export LZMA_VERSION
# export ZSTD_VERSION
# wget https://www.7-zip.org/a/lzma2301.7z -P $DLC
# wget https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz -P $DLC

if [ ! -f $DLC/elfutils-${ELF_VERSION}.tar.bz2 ]; then 
    wget https://sourceware.org/elfutils/ftp/${ELF_VERSION}/elfutils-${ELF_VERSION}.tar.bz2 -P $DLC
    tar xf $DLC/elfutils-${ELF_VERSION}.tar.bz2
    cd elfutils-${ELF_VERSION}
    ./configure --disable-libdebuginfod --disable-debuginfod --prefix=$ROOTDIR/installed
    make -j$NUM_VCORES
    make install
    cd -
fi

# -----------------------------------------------------------------------------
if true; then
    ./build-gcc.sh
fi

# -----------------------------------------------------------------------------
if true; then
    if [ ! -f $DLC/binutils-${BINUTILS_VERSION}.tar.xz ]; then 
        echo Downloading Binutils
        wget https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.xz -P $DLC
        tar xf $DLC/binutils-${BINUTILS_VERSION}.tar.xz
        cd binutils-${BINUTILS_VERSION}
        ./configure --prefix=$PWD/../installed
        make -j$NUM_VCORES
        make install 
        cd -
    fi
fi

# -----------------------------------------------------------------------------

if true; then
    if [ ! -f $DLC/Python-${PYTHON_VERSION}.tar.xz ];then
        wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz -P $DLC
        tar xf $DLC/Python-${PYTHON_VERSION}.tar.xz
        cd Python-${PYTHON_VERSION}
        time ./configure --prefix=${ROOTDIR}/installed --with-pydebug  --enable-optimizations #--with-lto
        time make -j$(nproc)
        time make install
        cd $ROOTDIR
    fi
fi

# -----------------------------------------------------------------------------
if true; then
    if [ ! -f $DLC/cmake-${CMAKE_VERSION}.tar.gz ];then
        wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz -P $DLC
        tar xf $DLC/cmake-${CMAKE_VERSION}.tar.gz
        cd cmake-${CMAKE_VERSION}
        ./bootstrap --prefix=$ROOTDIR/installed --parallel=$NUM_VCORES --no-qt-gui
        make -j$NUM_VCORES
        make install/strip
        cd $ROOTDIR
    fi
fi

# -----------------------------------------------------------------------------
if true; then
    if [ ! -f $DLC/re2c-${RE2C_VERSION}.tar.xz ];then
        wget https://github.com/skvadrik/re2c/releases/download/${RE2C_VERSION}/re2c-${RE2C_VERSION}.tar.xz -P $DLC
        tar xf $DLC/re2c-${RE2C_VERSION}.tar.xz 
        cd re2c-$RE2C_VERSION
        cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$ROOTDIR/installed
        cmake --build build --target install/strip
        cd $ROOTDIR
    fi
fi

# -----------------------------------------------------------------------------
# git clone git://github.com/ninja-build/ninja.git 
if true; then
    if [ ! -f $DLC/v${NINJA_VERSION}.tar.gz ];then
        wget https://github.com/ninja-build/ninja/archive/refs/tags/v${NINJA_VERSION}.tar.gz -P $DLC
        tar xf $DLC/v${NINJA_VERSION}.tar.gz
        cd ninja-$NINJA_VERSION
        cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$ROOTDIR/installed
        cmake --build build --target install/strip
        cd $ROOTDIR
    fi
fi

# -----------------------------------------------------------------------------
if [ ! -d jdk-11.0.22+7 ];then 
    wget -qO- https://github.com/adoptium/temurin11-binaries/releases/download/jdk-${JDK_VERSION}%2B${JDK_BUILD}/OpenJDK11U-jdk_x64_linux_hotspot_${JDK_VERSION}_${JDK_BUILD}.tar.gz | tar xzf -
fi

# -----------------------------------------------------------------------------
if true; then
    ./build-clang.sh
fi

if [ ! -f $DLC/$CUDA12_PACKAGE ];then
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/${CUDA12_PACKAGE} -P $DLC
fi

if [ ! -f $DLC/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz ];then
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$CUDNN9_PACKAGE -P $DLC
fi

if [ ! -d "$ROOTDIR/installed/cuda-$CUDA12_VERSION" ];then
  echo "Extracting CUDA $CUDA12_VERSION SDK and CUDNN9"
  chmod u+x "$DLC/$CUDA12_PACKAGE"
  "$DLC/$CUDA12_PACKAGE" --silent --toolkit --no-drm --no-man-page --no-opengl-libs --override --installpath="$ROOTDIR/installed/cuda-$CUDA12_VERSION"
  tar xf "$DLC/$CUDNN9_PACKAGE" --directory="$ROOTDIR/installed/cuda-$CUDA12_VERSION/targets/x86_64-linux/" --strip-components=1
fi

set +e
