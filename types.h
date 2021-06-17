#ifndef _TYPES_H
#define _TYPES_H

namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;

#define _TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH 1200
#define _TWOPOINT_FUNCTION_MAX_NUMBER_OF_GAMMAS 10
#define _TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA 12 

typedef struct {
  //number of gammas initially inside the twopoint function
  int number_of_gammas_source;
  int number_of_gammas_sink;
  int number_of_gammas_i1;
  int number_of_gammas_i2;
  int number_of_gammas_f1;
  int number_of_gammas_f2;
  // type, m-m, b-b, mxb-b, mxb-mxb
  char type [20];
  // name of 2-pt function
  char name[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  // doing the projection we have to identify the spin degrees of freedom
  // For nucleon it is 1
  // For the delta it is 3
  int contniuum_spin_particle_source;
  int contniuum_spin_particle_sink;
  // In the projection of interpolating operators we do not declare separately 
  // an operator for each gamma structure, but we include one with a list of gammas
  // Both for the baryon and for the meson 
  int list_of_gammas_f1[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_GAMMAS][2];
  int list_of_gammas_f2[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_GAMMAS];
  int list_of_gammas_i1[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_GAMMAS][2];
  int list_of_gammas_i2[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_GAMMAS];
  // number of diagrams
  int n;
  // list of diagrams
  char diagrams[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  // momenta at up to 4 vertices, default 0,0,0
  int pi1[3];
  int pi2[3];
  int pf1[3];
  int pf2[3];
  // For a definite total momentum, we have the freedom of choosing the
  // momenta of the pion and nucleon. These will provide us further
  // elements in the gevp
  // We have to define it for sink and source as well
  int total_momentum_pion_source[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA];
  int total_momentum_pion_sink[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA];
  int total_momentum_nucleon_source[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA];
  int total_momentum_nucleon_sink[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA];
  int ncombination_total_momentum_source;
  int ncombination_total_momentum_sink;
  // The list of momenta for pion and nucleon to give a specific values of total momentum
  int pf1list[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA][3];
  int pf2list[_TWOPOINT_FUNCTION_MAX_NUMBER_OF_MOMENTA][3];
  int nlistmomentumf1;
  int nlistmomentumf2;
  // gamma structure at up to 4 vertices, default -1
  int gi1[2];
  int gi2;
  int gf1[2];
  int gf2;
  // gamma matrix phases to match with used gamma basis; default is 1.
  double _Complex si1[2];
  double _Complex si2;
  double _Complex sf1[2];
  double _Complex sf2;
  // spin-1/2, spin-3/2 projection, default -1
  int spin_project;
  // parity projection, -1 / 0 [default] / +1
  int parity_project;
  // source coordinates, default -1,-1,-1,-1
  int source_coords[4];
  // reorder in time -1 / 0 [default] / +1
  int reorder;
  // normalization for diagrams
  char norm[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  // fwd / bwd propagation, default "NA"
  char fbwd[20];
  // number of timeslices, default -1, will be set to T_global
  int T;
  // spinor dimension, default 0
  int d;
  // data for this 2-pt function
  double _Complex  ****c;
  // group
  char group[100];
  // irrep
  char irrep[100];
  //name of the particles at the source and sink respectively
  char particlename_source[100]; 
  char particlename_sink[100];
} twopoint_function_type;

typedef struct {
  double *****v;
  double *ptr;
  int np;
  int ng;
  int nt;
  int nv;
} gsp_type;

}  /* end of namespace cvc */
#endif
