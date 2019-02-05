#ifndef _TYPES_H
#define _TYPES_H

namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;

#define _TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH 1200

typedef struct {
  // type, m-m, b-b, mxb-b, mxb-mxb
  char type [20];
  // name of 2-pt function
  char name[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  // number of diagrams
  int n;
  // list of diagrams
  char diagrams[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  // momenta at up to 4 vertices, default 0,0,0
  int pi1[3];
  int pi2[3];
  int pf1[3];
  int pf2[3];
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
