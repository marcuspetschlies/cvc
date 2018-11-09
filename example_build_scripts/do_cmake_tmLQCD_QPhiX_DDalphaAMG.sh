CXX=mpicxx \
CC=mpicc \
CXXFLAGS="-O3 -mtune=haswell -march=haswell -g -pg" \
CFLAGS="-O3 -mtune=haswell -march=haswell -g -pg" \
cmake \
  -DFFTW_HOME=/usr \
  -DLIME_HOME=/home/bartek/local/lime_gcc \
  -DDD_ALPHA_AMG_HOME=/home/bartek/build/DDalphaAMG \
  -DQPHIX_HOME=/home/bartek/local/qphix.devel/avx2 \
  -DQMP_HOME=/home/bartek/local/qmp \
  -DTMLQCD_SRC=/home/bartek/code/tmLQCD.etmc2 \
  -DTMLQCD_BUILD=/home/bartek/build/tmLQCD.etmc/build_hybrid_4D \
  ~/code/cvc.cpff
