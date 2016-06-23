#ifndef _TYPES_H
#define _TYPES_H
namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;

typedef struct {
  int gi;
  int gf;
  int pi[3];
  int pf[3];
} m_m_2pt_type;

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
