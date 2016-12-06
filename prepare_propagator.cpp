/************************************************
 * prepare_propagator.c
 *
 * Wed Feb  5 16:32:16 EET 2014
 *
 * PURPOSE:
 * - build propagator points
 ************************************************/

namespace cvc {

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
/* #include "smearing_techniques.h" */
/* #include "fuzz.h" */
#include "project.h"

#ifndef _NON_ZERO
#  define _NON_ZERO (5.e-14)
#endif

/************************************************************
 * prepare_seqprop_point_from_stochastic_oneend
 *   prepare propagator point from stochastic propagators
 *   output --- fp_out
 *   input  --- phi (stochastic, up), chi (stochastic, dn), prop (point source propagator),
 *              idsource, idsink (gamma matrix id at source and sink),
 *              ncol number of colors for the stochastic propagator
 *              momontum (sink momentum $\kvec_f$)
 *              N - number of sites
 * - NOTE that n_s = 4, n_c must be 3
 ************************************************************/
int prepare_seqprop_point_from_stochastic_oneend (double**fp_out, double **phi, double **chi,
    double **prop, const int idsource, int idsink, int ncol, double*phase_field, unsigned int N) {

  int n_s = 4;
  int n_c = 3;  // number of colors for the point propagator
  int psource[4], isimag, mu, c, ia, iprop;
  unsigned int ix;
  double ssource[4], spinor1[24], spinor2[24];
  complex w, w1, w2, ctmp;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];

  
  for(iprop=0; iprop<n_s*n_c; iprop++) {  // loop on spinor index of point source propagator
     
    for(mu=0; mu<4; mu++) {           // loop on spinor index of stochastic propagators
      for(c=0; c<ncol; c++) {         // loop on color  index of stochastic propagators

        // chi[psource[mu]]^dagger x exp(i k_f z_f) g5 x g_sink x prop[iprop]
        ctmp.re = 0.;
        ctmp.im = 0.;

        for(ix=0; ix<N; ix++) {
          _fv_eq_gamma_ti_fv(spinor1, idsink, prop[iprop]+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*ncol+c]+_GSI(ix), spinor2);
          w1.re = sin(phase_field[ix]);
          w1.im = cos(phase_field[ix]);
          _co_eq_co_ti_co(&w2, &w, &w1);
          ctmp.re += w2.re;
          ctmp.im += w2.im;
        }

        for(ix=0; ix<N; ix++) {

          for(ia=0; ia<n_c*n_s; ia++) {
            w.re = phi[mu*n_c+c][_GSI(ix) + 2*ia  ];
            w.im = phi[mu*n_c+c][_GSI(ix) + 2*ia+1];

            _co_eq_co_ti_co(&w1, &w, &ctmp);

            if( !isimag ) {
              fp_out[iprop][_GSI(ix) + 2*ia  ] +=  ssource[mu] * w1.re;
              fp_out[iprop][_GSI(ix) + 2*ia+1] +=  ssource[mu] * w1.im;
            } else {
              fp_out[iprop][_GSI(ix) + 2*ia  ] += -ssource[mu] * w1.im;
              fp_out[iprop][_GSI(ix) + 2*ia+1] +=  ssource[mu] * w1.re;
            }
          }
        }  // of loop on ix
      }    // of loop on colors for stochastic propagator
    }      // of loop on spin   for stochastic propagator
  }  // of loop on iprop
  return(0);
}  // end of prepare_seqprop_point_from_stochastic_oneend

/************************************************************
 * prepare_prop_point_from_stochastic
 *   prepare propagator point from stochastic sources
 *   and propagators
 *   - fp_out = phi chi^dagger
 *   - phi - propagator
 *   - chi - source
 * - NOTE this is without spin or color dilution
 ************************************************************/
int prepare_prop_point_from_stochastic (double**fp_out, double*phi, double*chi,
    double**prop, int idsink, double*phase_field, unsigned int N) {

  int n_s = 4;
  int n_c = 3;
  int ia, iprop;
  unsigned int ix;
  double spinor1[24];
  complex w, w1, w2, ctmp;

  for(iprop=0; iprop<12; iprop++) {

    // chi[psource[mu]]^dagger x exp(i k_f z_f) g5 x g_sink x prop[iprop]
    ctmp.re = 0.;
    ctmp.im = 0.;

    for(ix=0; ix<N; ix++) {
      _fv_eq_gamma_ti_fv(spinor1, idsink, prop[iprop]+_GSI(ix));
      _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
      w1.re = sin(phase_field[ix]);
      w1.im = cos(phase_field[ix]);
      _co_eq_co_ti_co(&w2, &w, &w1);
      ctmp.re += w2.re;
      ctmp.im += w2.im;
    }

    for(ix=0; ix<N; ix++) {

      for(ia=0; ia<n_s*n_c; ia++) {
        w.re = phi[_GSI(ix) + 2*ia  ];
        w.im = phi[_GSI(ix) + 2*ia+1];

        _co_eq_co_ti_co(&w1, &w, &ctmp);

        fp_out[iprop][2*ia  ] += w1.re;
        fp_out[iprop][2*ia+1] += w1.im;
      }
    }  // of loop on sites

  }    // of loop on iprop
  return(0);
}  // end of prepare_prop_point_from_stochastic

/*******************************************************************
 * seqn using stochastic
 *
 * shoulde be safe, if sfp = fp
 *
 * fp is T x VOL3 x g_fv_dim x g_fv_dim x 2
 *
 *******************************************************************/

int prepare_seqn_stochastic_vertex_propagator_sliced3d (fermion_propagator_type*sfp, double**stoch_prop, double**stoch_source, 
    fermion_propagator_type *fp, int nstoch, int momentum[3], int gid) {

  const unsigned int VOL3 = LX*LY*LZ;
  const size_t sizeof_spinor_field_timeslice = _GSI(VOL3) * sizeof(double);

  double *phase = (double*)malloc(2*VOL3*sizeof(double)) ;
  if( phase == NULL ) {
    fprintf(stderr, "[] Error from malloc\n");
    return(1);
  }
  make_lexic_phase_field_3d (phase,momentum);

  double **fpaux = (double**)malloc(g_fv_dim * sizeof(double*));
  fpaux[0] = (double*)malloc(g_fv_dim * sizeof_spinor_field_timeslice);
  if(fpaux[0] == NULL) {
    fprintf(stderr, "[] Error from malloc\n");
    return(2);
  }
  for(i=1; i<g_fv_dim; i++) fp_aux[i] = fp_aux[i-1] + _GSI(VOL3);

  for(it=0; it<T; it++) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int ix;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<VOL3; ix++) {
      _fv_eq_gamma_ti_fv()


    }  /* end of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  }  /* end of loop on timeslices */

  free(phase);
  free(fpaux[0]);
  free(fpaux);

  return(0);
}  /* prepare_seqn_stochastic_vertex_propagator_sliced3d */

}
