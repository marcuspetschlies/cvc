#ifndef _TYPES_H
#define _TYPES_H
namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;
typedef double *** b_1_xi_type;
typedef double *** w_1_xi_type;
typedef double **** b_all_phis_type;
typedef double **** w_all_phis_type;
typedef double **** V2_for_b_and_w_diagrams_type;

typedef struct {
  int gi;
  int gf;
  int pi[3];
  int pf[3];
} m_m_2pt_type;

typedef struct {
  int gi1;
  int gi2;
  int gf1;
  int gf2;
  int pi1[3];
  int pi2[3];
  int pf1[3];
  int pf2[3];
} mxb_mxb_2pt_type;

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
