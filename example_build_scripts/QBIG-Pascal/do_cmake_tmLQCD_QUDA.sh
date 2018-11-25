CXX=/opt/openmpi-2.0.2a1-with-pmi/bin/mpicxx \
CC=/opt/openmpi-2.0.2a1-with-pmi/bin/mpicc \
CXXFLAGS="-fopenmp -O3 -mtune=broadwell -march=broadwell -g" \
CFLAGS="-fopenmp -O3 -mtune=broadwell -march=broadwell -g" \
LDFLAGS="-fopenmp" \
cmake \
  -DMPI_HOME=/opt/openmpi-2.0.2a1-with-pmi \
  -DPARALLEL_LEVEL=TXYZ \
  -DQUDA_HOME=/qbigwork2/bartek/libs/bleeding_edge/pascal/quda_develop \
  -DLIME_HOME=/qbigwork2/bartek/libs/lime/broadwell \
  -DLEMON_HOME=/qbigwork2/bartek/libs/lemon/broadwell \
  -DTMLQCD_SRC=/qbigwork2/bartek/code/bleeding_edge/tmLQCD.get_cvc_to_work \
  -DTMLQCD_BUILD=/qbigwork2/bartek/build/bleeding_edge/pascal/tmLQCD.get_cvc_to_work.quda_develop \
  /qbigwork2/bartek/code/bleeding_edge/cvc.cpff
  
#-DLHPC_AFF_HOME=/qbigwork2/bartek/libs/broadwell \
#  -DGTEST_INCLUDE_DIRS=/hadron/bartek/local/gtest/include \

