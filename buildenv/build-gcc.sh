#!/usr/bin/env bash

source ./env.sh

if [ ! -f $DLC/gcc-${GCC_VERSION}.tar.xz ]; then
    wget ftp://ftp.lip6.fr/pub/gcc/releases/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz -P $DLC
fi

tar xf $DLC/gcc-${GCC_VERSION}.tar.xz
cd gcc-${GCC_VERSION}
./contrib/download_prerequisites
mkdir ../build-gcc
cd ../build-gcc
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CFLAGS CXXFLAGS
../gcc-${GCC_VERSION}/configure --prefix=$ROOTDIR/installed --enable-languages=c,c++,fortran --disable-libquadmath \
     --disable-lto --disable-libquadmath-support --disable-werror --disable-bootstrap --enable-gold --disable-multilib --disable-libssp
make -j$NUM_VCORES
make install
cd $ROOTDIR/installed/bin
ln -sf gcc cc
ln -sf g++ c++
cd $ROOTDIR

