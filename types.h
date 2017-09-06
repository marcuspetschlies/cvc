#ifndef _TYPES_H
#define _TYPES_H
namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;

typedef struct {
  char type [20];
  char name[50];
  int n;
  char diagrams[400];
  int pi1[3];
  int pi2[3];
  int pf1[3];
  int pf2[3];
  int gi1[2];
  int gi2;
  int gf1[2];
  int gf2;
  int spin_project;
  int parity_project;
  int source_coords[4];
  int reorder;
  char norm[500];
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
