#!/bin/bash
. load_modules.sh
CXX=CC \
CC=cc \
CXXFLAGS="-fopenmp -O3 -mtune=haswell -march=haswell -g" \
CFLAGS="-fopenmp -O3 -mtune=haswell -march=haswell -g" \
LDFLAGS="-fopenmp" \
cmake \
  -DPARALLEL_LEVEL=TXYZ \
  -DLAPACK_HOME=/opt/cray/pe/libsci/18.07.1/GNU/6.1/x86_64 \
  -DLAPACK_LIBRARIES="-lsci_gnu_mp" \
  -DBLAS_HOME=/opt/cray/pe/libsci/18.07.1/GNU/6.1/x86_64 \
  -DBLAS_LIBRARIES="-lsci_gnu_mp" \
  -DQUDA_HOME=/project/s849/bartek/libs/2018_11_19/quda_develop-dynamic_clover \
  -DLIME_HOME=/users/bartek/local/haswell/libs/lime \
  -DLEMON_HOME=/users/bartek/local/haswell/libs/lemon \
  -DTMLQCD_SRC=/users/bartek/code/2018_11_19/tmLQCD.get_cvc_to_work \
  -DTMLQCD_BUILD=/project/s849/bartek/build/2018_11_19/tmLQCD.get_cvc_to_work.quda_develop-dynamic_clover \
  /users/bartek/code/2018_11_19/cvc_cpff

# if AFF output is desired
#  -DLHPC_AFF_HOME=/users/bartek/local/haswell/libs/lhpc-aff \

