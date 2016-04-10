#!/bin/sh
set -x

DIR=caffe

# change default caffe directory
if [ -n "$1" ] ; then
	DIR=$1
fi

cd $DIR
rm -fr build
mkdir build
cd build
cmake ../   -DAtlas_LAPACK_LIBRARY=/usr/lib64/libatlas.so \
	-DAtlas_BLAS_LIBRARY=/usr/lib64/libatlas.so \
	-DCMAKE_INSTALL_PREFIX=/usr/local/

make -j16 all
