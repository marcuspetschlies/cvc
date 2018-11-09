CXX=/opt/openmpi-2.0.2a1-with-pmi/bin/mpicxx \
CC=/opt/openmpi-2.0.2a1-with-pmi/bin/mpicc \
CXXFLAGS="-O3 -mtune=sandybridge -march=sandybridge -g -pg" \
CFLAGS="-O3 -mtune=sandybridge -march=sandybridge -g -pg" \
cmake \
  -DMPI_HOME=/opt/openmpi-2.0.2a1-with-pmi \
  -DPARALLEL_LEVEL=TXYZ \
  -DLHPC_AFF_HOME=/qbigwork2/bartek/libs/sandybridge \
  -DQUDA_HOME=/qbigwork2/bartek/libs/bleeding_edge/kepler/quda_develop \
  -DLIME_HOME=/qbigwork2/bartek/libs/lime/sandybridge \
  -DLEMON_HOME=/qbigwork2/bartek/libs/lemon/sandybridge \
  -DTMLQCD_SRC=/qbigwork2/bartek/code/bleeding_edge/tmLQCD.test_cvc_cpff \
  -DTMLQCD_BUILD=/qbigwork2/bartek/build/bleeding_edge/kepler/tmLQCD.quda_develop.test_cvc_cpff \
  -DFFTW_HOME=/usr \
  /qbigwork2/bartek/code/bleeding_edge/cvc.cpff
