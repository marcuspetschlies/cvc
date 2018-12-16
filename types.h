#ifndef _TYPES_H
#define _TYPES_H

#include "enums.hpp"

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

typedef struct shift_t {
  int origin_shift[4];
  latDim_t dim;
  shift_dir_t dir;
} shift_t;

}  /* end of namespace cvc */
#endif
