# ##########
work=/home/marcuspe/software/
srcdir=..

qlua_build_dir=$(work)/qlua/momproj/build-ubuntu18/

limedir=$(work)/lime-1.3.2/build-ubuntu18/

lemondir=$(work)/lemon/build-ubuntu18/

fftwdir=$(work)/fftw/build-ubuntu18

lapackdir=$(qlua_build_dir)/lapack

lhpcdir=$(qlua_build_dir)/aff

hdf5_include_dir=/usr/include/hdf5/serial

CXX=g++ -fopenmp
CXXFLAGS= -Wall -g -O2 -DF_ -DHAVE_LHPC_AFF -DHAVE_OPENMP  -DCUBIC_GROUP_SINGLE_COVER -DDUMMY_SOLVER -DHAVE_HDF5
# -DCUBIC_GROUP_DOUBLE_COVER


CCDEP = g++
DEPFLAGS = -MM

INCLUDE = -I$(limedir)/include/ -I$(srcdir) -I$(fftwdir)/include -I. -I$(lhpcdir)/include -I$(hdf5_include_dir)/

LIBS = -L$(lhpcdir)/lib  -L$(limedir)/lib/ -L$(fftwdir)/lib -lfftw_mpi -lfftw -llime -llapack -lblas -lgfortran -lm -llhpc-aff -lhdf5_serial

LDFLAGS = -fopenmp -L$(arpackdir)/lib -L$(limedir)/lib/ -L$(fftwdir)/lib -lfftw_mpi -lfftw -llime -llapack -lblas -lgfortran -lm -llhpc-aff -lhdf5_serial


LINK = $(CXX) -o $@ ${LDFLAGS}
COMPILE = ${CXX} $(INCLUDE) -o $@ ${CXXFLAGS}

MODULES = DML_crc32 dml getopt cvc_utils cvc_geometry mpi_init io io_utils propagator_io read_input_parser_cvc \
	  get_index gauge_io contractions_io ranlxd ranlxs Q_phi set_default \
	  Q_clover_phi gsp prepare_source matrix_init project make_x_orbits contract_cvc_tensor smearing_techniques \
	  prepare_propagator scalar_products clover rotations gitversion group_projection gamma dummy_solver contract_factorized \
	  contract_loop uwerr incomp_gamma derived_quantities

HEADERS = getopt cvc_complex cvc_geometry cvc_linalg iblas icontract cvc_utils default_input_values dml global io io_utils \
	  mpi_init propagator_io read_input_parser get_index gauge_io contractions_io \
	  ranlxd ranlxs Q_phi set_default laplace_linalg Q_clover_phi \
	  gsp prepare_source matrix_init project make_x_orbits contract_cvc_tensor smearing_techniques \
	  prepare_propagator scalar_products clover rotations group_projection \
          set_cubic_group_double_cover_elements set_cubic_group_double_cover_identification_table set_cubic_group_elements \
          set_cubic_group_elements_v2 gamma gamma_mult_table clebsch_gordan \
	  table_init_z table_init_d table_init_i table_init_c table_init_asym_z group_projection_applicator dummy_solver contract_factorized \
	  contract_loop uwerr incomp_gamma derived_quantities


PROGRAM = cpff_invert_contract 



all: dep $(PROGRAM) 


# ##########

read_input_parser_cvc.cpp: ${srcdir}/read_input_parser_cvc.l
	${LEX} -P cvc_ -i -t $< > ${srcdir}/read_input_parser_cvc.cpp

$(addsuffix .d,$(MODULES)): %.d: ${srcdir}/%.cpp
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE} $< > $@

$(addsuffix .d,$(PROGRAM)): %.d: ${srcdir}/%.cpp
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE} $< > $@

dep: $(addsuffix .d,$(MODULES) ${PROGRAM})

$(addsuffix .o,${MODULES}): %.o: ${srcdir}/%.cpp $(addprefix ${srcdir}/, $(addsuffix .h, ${HEADERS})) %.d
	${COMPILE} ${OPTARGS} -c $< 

$(addsuffix .o,${PROGRAM}): %.o: ${srcdir}/%.cpp %.d
	${COMPILE} ${OPTARGS} -c $< 

${PROGRAM}: %: %.o gitversion.cpp $(addsuffix .o,${MODULES})
	${LINK}  $(addsuffix .o,${MODULES}) $@.o $(LIBS)

gitversion.cpp: ${srcdir}/.git/HEAD ${srcdir}/.git/index
	echo "namespace cvc { const char *g_gitversion = \"$(shell cd ${srcdir} && git rev-parse HEAD && cd -)\"; }" > ${srcdir}/gitversion.cpp

# ##########


clean:
	rm -f *~ *.o *.d $(PROGRAM) ${srcdir}/gitversion.cpp

.PHONY: clean

# ##########
