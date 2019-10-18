/****************************************************
 * contract_diagrams.c
 * 
 * Mon Jun  5 16:00:53 CDT 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif
#ifdef HAVE_HDF5
#include "hdf5.h"
#endif

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "table_init_z.h"
#include "gamma.h"
#include "zm4x4.h"
#include "rotations.h"
#include "contract_diagrams.h"

namespace cvc {

#if 0
/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, beta]  v3[beta,m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[3], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {

          int idx[3] = { alpha, beta, gamma };

          int pidx[3] = { idx[perm[0]], idx[perm[1]], idx[perm[2]] };

          for ( int delta = 0; delta < 4; delta++ ) {
            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * delta + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

}  /* end of function contract_diagram_v2_gamma_v3 */
#endif  /* of if 0*/

/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[ alpha_p[3], m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int const perm[4], unsigned int const N, int const init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [contract_diagram_v2_gamma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {
          for ( int delta = 0; delta < 4; delta++ ) {

            int idx[4]  = { alpha, beta, gamma, delta };

            int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * pidx[3] + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

  return(0);
}  /* end of function contract_diagram_v2_gamma_v3 */

/****************************************************
 * we always sum in the following way
 * goet[b_oet][a_oet]  v2[a_oet][alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[b_oet][ alpha_p[3], m]
 ****************************************************/
int contract_diagram_oet_v2_gamma_v3 ( double _Complex **vdiag, double _Complex ***v2, double _Complex ***v3, gamma_matrix_type goet, gamma_matrix_type g, int const perm[4], unsigned int const N, int const init ) {

  if ( init ) {
    if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_oet_v2_amma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

  for ( int sigma_oet = 0; sigma_oet < 4; sigma_oet++ ) {
  for ( int tau_oet   = 0; tau_oet   < 4; tau_oet++ ) {

    double _Complex c = goet.m[tau_oet][sigma_oet];
    if ( c == 0 ) continue;

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int it = 0; it < N; it++ ) {

      for ( int alpha = 0; alpha < 4; alpha++ ) {
        for ( int beta = 0; beta < 4; beta++ ) {

          int vdiag_index = 4 * alpha + beta;
          /* vdiag[it][vdiag_index] = 0.; */

          /****************************************************/
          /****************************************************/

          for ( int gamma = 0; gamma < 4; gamma++ ) {
            for ( int delta = 0; delta < 4; delta++ ) {

              int idx[4]  = { alpha, beta, gamma, delta };

              int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

              for ( int m = 0; m < 3; m++ ) {

                /* use the permutation */
                int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
                int v3_index  = 3 * pidx[3] + m;

                vdiag[it][vdiag_index] -=  c * v2[sigma_oet][it][v2_pindex] * v3[tau_oet][it][v3_index] * g.m[gamma][delta];
              }  /* end of loop on color index m */
            }  /* end of loop on spin index delta */
          }  /* end of loop on spin index gamma */

          /****************************************************/
          /****************************************************/

        }  /* end of loop on spin index beta */
      }  /* end of loop on spin index alpha */

    }  /* end of loop on N */

  }  /* end of loop on tau   oet */
  }  /* end of loop on sigma oet */
  return(0);
}  /* end of function contract_diagram_oet_v2_gamma_v3 */

#if 0
/****************************************************
 *
 ****************************************************/
void contract_b1 (double _Complex ***b1, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + beta ) + alpha ) + delta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b1[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b1 */

void contract_b2 (double _Complex ***b2, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + delta ) + alpha ) + beta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b2[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b2 */
#endif  /* end of if 0 */

/****************************************************
 * search for m1 in m2
 ****************************************************/
int match_momentum_id ( int **pid, int **m1, int **m2, int N1, int N2 ) {

  if ( g_verbose > 4 && g_cart_id == 0 ) {
    fprintf(stdout, "# [match_momentum_id] N1 = %d N2 = %d m2 == NULL ? %d\n", N1, N2 , m2 == NULL);
    for ( int i = 0; i < N1; i++ ) {
      fprintf(stdout, "# [match_momentum_id] m1 %d  %3d %3d %3d\n", i, m1[i][0], m1[i][1], m1[i][2]);
    }

    for ( int i = 0; i < N2; i++ ) {
      fprintf(stdout, "# [match_momentum_id] m2 %d  %3d %3d %3d\n", i, m2[i][0], m2[i][1], m2[i][2]);
    }
  }

  /****************************************************/
  /****************************************************/

  /* if ( N1 > N2 ) {
    fprintf(stderr, "[match_momentum_id] Error, N1 > N2\n");
    return(1);
  } */

  if ( *pid == NULL ) {
    if ( ( *pid = (int*)malloc (N1 * sizeof(int) ) ) == NULL ) {
      fprintf(stderr, "# [match_momentum_id] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(4);
    }
  }

  int found_no_match = 1;

  for ( int i = 0; i < N1; i++ ) {
    int found = 0;
    int p[3] = { m1[i][0], m1[i][1], m1[i][2] };

    for ( int k = 0; k < N2; k++ ) {
      if ( p[0] == m2[k][0] && p[1] == m2[k][1] && p[2] == m2[k][2] ) {
        (*pid)[i] = k;
        found = 1;
        break;
      }
    }
    if ( found == 0 ) {
      fprintf(stderr, "[match_momentum_id] Warning, could not find momentum no %d = %3d %3d %3d\n",
          i, p[0], p[1], p[2]);
      (*pid)[i] = -1;
      /* return(2); */
    }

    found_no_match = found_no_match && !found;

  }  // end of loop on momenta to be matched

  // TEST
  if ( g_verbose > 2 ) {
    for ( int i = 0; i < N1; i++ ) {
      if ( (*pid)[i] == -1 ) {
        fprintf(stdout, "# [match_momentum_id] m1[%2d] = %3d %3d %3d no match\n", i, m1[i][0], m1[i][1], m1[i][2] );
      } else {
        fprintf(stdout, "# [match_momentum_id] m1[%2d] = %3d %3d %3d matches m2[%2d] = %3d %3d %3d\n",
            i, m1[i][0], m1[i][1], m1[i][2],
            (*pid)[i], m2[(*pid)[i]][0], m2[(*pid)[i]][1], m2[(*pid)[i]][2]);
      }
    }
  }

  return( found_no_match );
}  // end of match_momentum_id

/***********************************************/
/***********************************************/

/***********************************************
 * p1 == g_seq2_source_momentum_list
 * p2 == g_total_momentum vector
 * p3 == g_sink_momentum_list
 *
 ***********************************************/

int * get_conserved_momentum_id ( int (*p1)[3], int const n1, int const p2[3], int (*p3)[3], int const n3 ) {

  int exitstatus;
  int **momentum_list = NULL, **momentum_list_all = NULL;

  exitstatus = init_2level_ibuffer ( &momentum_list, n1, 3 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[get_minus_momentum_id] Error from init_2level_ibuffer, status was %d\n", exitstatus );
    return( NULL );
  }
   
  exitstatus = init_2level_ibuffer ( &momentum_list_all, n3 , 3 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[get_minus_momentum_id] Error from init_2level_ibuffer, status was %d\n", exitstatus );
    return ( NULL );
  }

  for ( int i = 0; i < n3; i++ ) {
    momentum_list_all[i][0] = p3[i][0];
    momentum_list_all[i][1] = p3[i][1];
    momentum_list_all[i][2] = p3[i][2];
  }
   
  for ( int i = 0; i < n1; i++ ) {
    momentum_list[i][0] = p2[0] - p1[i][0];
    momentum_list[i][1] = p2[1] - p1[i][1];
    momentum_list[i][2] = p2[2] - p1[i][2];
  }
 
  int *momentum_id = NULL;
  exitstatus = match_momentum_id ( &momentum_id, momentum_list, momentum_list_all, n1, n3 );
  if ( exitstatus == 1 ) {
    fprintf(stderr, "[get_minus_momentum_id] Error, not a single match\n");
    free ( momentum_id );
    return ( NULL );
  } else if ( exitstatus != 0 ) {
    fprintf(stderr, "[get_minus_momentum_id] Error from match_momentum_id, status was %d\n", exitstatus );
    return ( NULL );
  }
  
  fini_2level_ibuffer ( &momentum_list );
  fini_2level_ibuffer ( &momentum_list_all );

  return( momentum_id );

}  // end of get_conserved_momentum_id

/***********************************************
 *
 ***********************************************/
int get_momentum_id ( int const p1[3], int (* const p2)[3], unsigned int const N ) {
  for ( unsigned int i = 0; i < N; i++ ) {
    if ( ( p1[0] == p2[i][0] ) && ( p1[1] == p2[i][1] ) && ( p1[2] == p2[i][2] ) ) {
       return ( i );   
    }

  }
  return ( -1 );
}  // end of get_momentum_id

/***********************************************/
/***********************************************/

/***********************************************
 * multiply spinor propagator field
 *   with boundary phase
 ***********************************************/
int correlator_add_baryon_boundary_phase ( double _Complex *** const sp, int const tsrc, int const dir, int const N ) {

  if( g_propagator_bc_type == 0 ) {
    // multiply with phase factor
    if ( g_verbose > 3 ) fprintf(stdout, "# [correlator_add_baryon_boundary_phase] multiplying with boundary phase factor\n");

    if ( tsrc > 0 ) {
      // assume lattice ordering
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( int it = 0; it < N; it++ ) {
        int ir = (it + g_proc_coords[0] * T - tsrc + T_global) % T_global;
        const double _Complex w = cexp ( I * 3. * M_PI*(double)ir / (double)T_global  );
        zm4x4_ti_eq_co ( sp[it], w );
      }
    } else if ( tsrc == 0 ) {
      // assume ordering from source

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( int it = 0; it < N; it++ ) {
        const double _Complex w = cexp ( I * 3. * M_PI*(double)( (dir*it + T_global ) % T_global )  / (double)T_global  );
        zm4x4_ti_eq_co ( sp[it], w );
      }
    }

  } else if ( g_propagator_bc_type == 1 ) {

    // multiply with step function
    if ( g_verbose > 3 ) fprintf(stdout, "# [add_baryon_boundary_phase] multiplying with boundary step function\n");


    // MUST assume lattice ordering timeslices 0, 1, ..., T-1

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for( int ir = 0; ir < N; ir++ ) {                            // counter of global timeslice
      int const it = ( dir * ir + tsrc + T_global ) % T_global;  // global timeslice, 0 <= it < T_global
      int const is = it % T;                                     // local timeslice, 0 <= is < T
#ifdef HAVE_MPI
      //    need -1           my timeslice
      if( ( it < tsrc ) && ( it / T == g_proc_coords[0] ) ) 
#else
      if( ( it < tsrc ) )
#endif
      {

        zm4x4_ti_eq_re ( sp[is], -1 );

      }  // end of if it < tsrc

    }  // end of loop on ir

  }  // end of if propagator bc type is 1

  return(0);
}  // end of correlator_add_baryon_boundary_phase


/***********************************************
 * multiply with phase from source location
 * - using pi1 + pi2 = - ( pf1 + pf2 ), so
 *   pi1 = - ( pi2 + pf1 + pf2 )
 ***********************************************/
int correlator_add_source_phase ( double _Complex ***sp, int const p[3], int const source_coords[3], unsigned int const N ) {

  const double _Complex TWO_MPI_I = 2. * M_PI * I;

  const double _Complex w = cexp ( TWO_MPI_I * ( ( p[0] / (double)LX_global ) * source_coords[0] + 
                                                 ( p[1] / (double)LY_global ) * source_coords[1] + 
                                                 ( p[2] / (double)LZ_global ) * source_coords[2] ) );
  
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) {
#ifdef HAVE_MPI
    // not my timeslice ? go on
    if ( ix / T != g_proc_coords[0] ) continue;
    unsigned int const is = ix % T;
#else
    unsigned int const is = ix;
#endif
    zm4x4_ti_eq_co ( sp[is], w );
  }
  return(0);
}  /* end of correlator_add_source_phase */

int correlator_spin_projection (double _Complex ***sp_out, double _Complex ***sp_in, int const i, int const k, double const a, double const b, unsigned int const N) {

  int ik = 4*i+k;
  
  switch(ik) {
    case  5: 
    case 10: 
    case 15: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_33 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  6: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_12 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  7: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_13 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  9: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_21 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 11: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_23 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 13: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_31 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 14: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_32 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    default:
      fprintf(stderr, "[correlator_spin_projection] Error, projector P_{%d,%d} not implemented\n", i, k);
      return(1);
      break;;
  }  /* end of switch i, k */
  return(0);
}  /* end of correlator_spin_projection */

/***********************************************
 *
 ***********************************************/
int correlator_spin_parity_projection (double _Complex ***sp_out, double _Complex ***sp_in, double const c, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_spin_parity_projection_zm4x4 ( sp_out[ir], sp_in[ir], c);
  }
  return(0);
}  /* end of correlator_spin_parity_projection */

/***********************************************/
/***********************************************/

/***********************************************
 *
 ***********************************************/
int reorder_to_absolute_time (double _Complex ***sp_out, double _Complex ***sp_in, int const tsrc, int const dir, unsigned int const N) {

  int exitstatus;
  double _Complex ***buffer = NULL;

  if ( dir == 0 ) {
    // nothing to be done except if sp_out and sp_in are not the same fields in memory
    if ( sp_out != sp_in ) {
      memcpy( sp_out[0][0], sp_in[0][0],  N*16*sizeof(double _Complex) );
    }
    return(0);
  }

  exitstatus = init_3level_zbuffer ( &buffer, N, 4, 4);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reorder_to_absolute_time] Error from init_3level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  memcpy( buffer[0][0], sp_in[0][0],  N*16*sizeof(double _Complex) );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    unsigned int is = ( ir + ( dir * tsrc + N ) ) % N;
    zm4x4_eq_zm4x4( sp_out[is], buffer[ir] );
  }
  fini_3level_zbuffer ( &buffer );
  return(0);
}  /* end of reorder_to_absolute_time */

/***********************************************
 * wrapper for reordering to absolute,
 * which uses dir = +1
 ***********************************************/
int reorder_to_relative_time (double _Complex ***sp_out, double _Complex ***sp_in, int const tsrc, int const dir, unsigned int const N) {
  return ( reorder_to_absolute_time ( sp_out, sp_in, tsrc, -dir, N) );
}  /* end of reorder_to_relative_time */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_ti_co_field ( double _Complex ***sp_out, double _Complex ***sp_in, double _Complex *c_in, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_zm4x4_ti_co ( sp_out[ir], sp_in[ir], c_in[ir] );
#if 0
    if ( ir == 0 && g_verbose > 3 ) {
      zm4x4_printf ( sp_in[ir], "sp_in", stdout );
      fprintf(stdout, "# [contract_diagram_zm4x4_field_ti_co_field] c_in = %25.15e %25.16e\n", creal(c_in[ir]), cimag(c_in[ir]));
    }
#endif  /* of if 0 */
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_ti_co_field */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_eq_zm4x4_field_ti_co ( double _Complex *** const sp_out, double _Complex *** const sp_in, double _Complex const c_in, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_zm4x4_ti_co ( sp_out[ir], sp_in[ir], c_in );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_eq_zm4x4_field_ti_co */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_ti_eq_co ( double _Complex *** const sp_out, double _Complex const c_in, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_zm4x4_ti_co ( sp_out[ir], sp_out[ir], c_in );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_ti_eq_co */



/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_ti_eq_re ( double _Complex *** const sp, double const r_in, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_ti_eq_re ( sp[ir], r_in );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_ti_eq_re */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( double _Complex *** const s, double _Complex *** const r, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_pl_eq_zm4x4 ( s[ir], r[ir] );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_pl_eq_zm4x4 */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co ( double _Complex *** const s, double _Complex *** const r, double _Complex *** const p, double _Complex const z, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_zm4x4_pl_zm4x4_ti_co ( s[ir], r[ir], p[ir], z );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co */

/***********************************************/
/***********************************************/

int contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( double _Complex *** const sp_out, double _Complex *** const sp_in, unsigned int const N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ir = 0; ir < N ; ir++ ) {
    zm4x4_eq_zm4x4_transposed ( sp_out[ir], sp_in[ir] );
  }
  return(0);
}  /* end of contract_diagram_zm4x4_field_eq_zm4x4_field_transposed */

/***********************************************/
/***********************************************/

/***********************************************
 *
 ***********************************************/
int contract_diagram_sample (double _Complex ***diagram, double _Complex ***xi, double _Complex ***phi, int const nsample, int const perm[4], gamma_matrix_type C, int const nT ) {

  int exitstatus;
  double _Complex **diagram_buffer = NULL;
  double _Complex znorm = 1. / (double)nsample;


  /***********************************************
   * allocate diagram_buffer to accumulate
   *
   * initialized to 0
   ***********************************************/
  if ( ( exitstatus= init_2level_zbuffer ( &diagram_buffer, nT, 16 ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }


  for ( int isample = 0; isample < nsample; isample++ ) {
    if ( ( exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, phi[isample], xi[isample], C, perm, nT, (int)(isample==0) ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_sample] Error from contract_diagram_v2_gamma_v3, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(1);
    }
  }
 
  /* copy to diagram */  
  memcpy( diagram[0][0], diagram_buffer[0], 16*nT*sizeof(double _Complex) );

  /* transpose (to conform with full diagram code) */
  if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, nT ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  /* normalize with 1 / nsample */
  if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, znorm, nT ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /* clean up diagram_buffer */
  fini_2level_zbuffer ( &diagram_buffer );
  return(0);
}  /* end of contract_diagram_sample */

/***********************************************/
/***********************************************/

/***********************************************
 *
 ***********************************************/
int contract_diagram_sample_oet (double _Complex ***diagram, double _Complex ***xi, double _Complex ***phi, gamma_matrix_type goet, int const perm[4], gamma_matrix_type C, int const nT ) {

  int exitstatus;
  double _Complex **diagram_buffer = NULL;

  if ( ( exitstatus= init_2level_zbuffer ( &diagram_buffer, nT, 16 ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample_oet] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  if ( ( exitstatus = contract_diagram_oet_v2_gamma_v3 ( diagram_buffer, phi, xi, goet, C, perm, nT, 1 ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample_oet] Error from contract_diagram_oet_v2_gamma_v3, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /* copy to diagram */  
  memcpy( diagram[0][0], diagram_buffer[0], 16*nT*sizeof(double _Complex) );

  /* transpose (to conform with full diagram code) */
  if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, nT ) ) != 0 ) {
    fprintf(stderr, "[contract_diagram_sample_oet] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  /* clean up diagram_buffer */
  fini_2level_zbuffer ( &diagram_buffer );
  return(0);
}  /* end of contract_diagram_sample_oet */

/***********************************************/
/***********************************************/

#ifdef HAVE_LHPC_AFF
/***********************************************
 * write contracted diagram to AFF
 *
 * index sequence t - s_sink - s_source
 ***********************************************/
int contract_diagram_write_aff (double _Complex***diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc ) {

  const unsigned int offset = 16;
  const size_t bytes = offset * sizeof(double _Complex);
  const int nt = dt + 1; /* tstart + dt */

  int exitstatus;
  double rtime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  double _Complex ***aff_buffer = NULL;

  if ( io_proc == 2 ) {
    rtime = _GET_TIME;

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_diagram_write_aff] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    if( ( exitstatus = init_3level_zbuffer ( &aff_buffer, nt, 4, 4 ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_write_aff] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }

    for ( int i = 0; i <= dt; i++ ) {
      int t = ( tstart + i * fbwd  + T_global ) % T_global;
      memcpy( aff_buffer[i][0], diagram[t][0], bytes );
    }

    affdir = aff_writer_mkpath (affw, affn, aff_tag );
    if ( ( exitstatus = aff_node_put_complex (affw, affdir, aff_buffer[0][0], (uint32_t)(nt*offset) ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_write_aff] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      return(1);
    }

    fini_3level_zbuffer ( &aff_buffer );

    rtime = _GET_TIME - rtime;
    if (g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_write_aff] time for contract_diagram_write_aff = %e seconds %s %d\n", rtime, __FILE__, __LINE__);
  }  /* end of if io_proc == 2 */
  return(0);
}  /* end of contract_diagram_write_aff */


/***********************************************
 * write contracted diagram to AFF
 *
 * index sequence s_sink - s_source - t
 ***********************************************/
int contract_diagram_write_aff_sst (double _Complex***diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc ) {

  const unsigned int offset = 16;
  const size_t bytes = offset * sizeof(double _Complex);
  const int nt = dt + 1; /* tstart + dt */

  int exitstatus;
  double rtime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  double _Complex ***aff_buffer = NULL;

  if ( io_proc == 2 ) {
    rtime = _GET_TIME;

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_diagram_write_aff_sst] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    if( ( exitstatus = init_3level_zbuffer ( &aff_buffer, 4, 4, nt ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_write_aff_sst] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }

#pragma omp parallel for
    for ( int i = 0; i <= dt; i++ ) {
      int const t = ( tstart + i * fbwd  + T_global ) % T_global;
      for ( int isl = 0; isl < 4; isl++ ) {
      for ( int isr = 0; isr < 4; isr++ ) {
        aff_buffer[isl][isr][i] = diagram[t][isl][isr];
      }}
    }  /* end of loop on timeslices */

    affdir = aff_writer_mkpath (affw, affn, aff_tag );
    if ( ( exitstatus = aff_node_put_complex (affw, affdir, aff_buffer[0][0], (uint32_t)(nt*offset) ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_write_aff_sst] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      return(1);
    }

    fini_3level_zbuffer ( &aff_buffer );

    rtime = _GET_TIME - rtime;
    if (g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_write_aff_sst] time for contract_diagram_write_aff_sst = %e seconds %s %d\n", rtime, __FILE__, __LINE__);
  }  /* end of if io_proc == 2 */
  return(0);
}  /* end of contract_diagram_write_aff_sst */
#endif  // of if def HAVE_LHPC_AFF

/***********************************************/
/***********************************************/
#if 0
#ifdef HAVE_HDF5
/***********************************************
 * write contracted diagram to HDF5
 ***********************************************/
int contract_diagram_write_h5 (double _Complex***diagram, hid_t file, hid_t memtype, hid_t filetype, char * tag, int const tstart, int const dt, int const fbwd, int const io_proc ) {

  const unsigned int offset = 16;
  const size_t bytes = offset * sizeof(double _Complex);
  const int nt = dt + 1; // tstart + dt

  int exitstatus;
  double rtime;
  herr_t status;

  if ( io_proc == 2 ) {
    rtime = _GET_TIME;

    complex *wbuffer = (complex*)malloc ( nt*16*sizeof(complex) );
    if ( wbuffer == NULL ) {
      fprintf(stderr, "[contract_diagram_write_h5] Error from wbuffer %s %d\n", __FILE__, __LINE__);
     return(1);
    }

    // extract data for tstart <= t <= tstart + dt
    for ( int i = 0; i <= dt; i++ ) {
      int t = ( tstart + i * fbwd  + T_global ) % T_global;
      memcpy( wbuffer + i*offset, diagram[t][0], bytes );
    }

    //     hid_t       file, filetype, memtype, strtype, space, dset;
    // create dataspace
    int dims[1] = { offset*nt };
    hid_t space = H5Screate_simple (1, dims, NULL);
    if ( space < 0 ) {
      fprintf ( stderr, "# [contract_diagram_write_h5] Error from H5Screate_simple, status was %d %s %d\n", status, __FILE__, __LINE__ );
      return (2);
    }

    // create dataset
    hid_t dset = H5Dcreate ( file, dataset, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    if ( dset < 0 ) {
      fprintf ( stderr, "# [contract_diagram_write_h5] Error from H5Dcreat, status was %d %s %d\n", status, __FILE__, __LINE__ );
      return (3);
    }

    // write data to file
    status = H5Dwrite ( dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuffer );
    if ( status < 0 ) {
      fprintf ( stderr, "# [contract_diagram_write_h5] Error from H5Dwrite, status was %d %s %d\n", status, __FILE__, __LINE__ );
      return (4);
    }


    free( wbuffer );
    status = H5Sclose (space);
    status = H5Dclose (dset);

    rtime = _GET_TIME - rtime;
    if (g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_write_h5] time for contract_diagram_write_h5 = %e seconds %s %d\n", rtime, __FILE__, __LINE__);
  }  // end of if io_proc == 2
  return(0);
}  // end of contract_diagram_write_h5
#endif  // of if def HAVE_HDF5
#endif

/***********************************************/
/***********************************************/

/***********************************************
 * write contracted scalar diagram to AFF
 ***********************************************/
#ifdef HAVE_LHPC_AFF
int contract_diagram_write_scalar_aff (double _Complex*diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc ) {

  const int nt = dt + 1; /* tstart + dt */

  int exitstatus;
  double rtime;
  struct AffNode_s *affn = NULL, *affdir=NULL;

  if ( io_proc == 2 ) {
    rtime = _GET_TIME;

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_diagram_write_scalar_aff] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    double _Complex *aff_buffer = init_1level_ztable ( nt );
    if ( aff_buffer == NULL ) {
      fprintf(stderr, "[contract_diagram_write_scalar_aff] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__);
      return(6);
    }

    for ( int i = 0; i <= dt; i++ ) {
      int t = ( tstart + i * fbwd  + T_global ) % T_global;
      aff_buffer[i] = diagram[t];
    }

    affdir = aff_writer_mkpath (affw, affn, aff_tag );
    uint32_t uitems = (uint32_t)nt;
    if ( ( exitstatus = aff_node_put_complex (affw, affdir, aff_buffer, uitems ) ) != 0 ) {
      fprintf(stderr, "[contract_diagram_write_scalar_aff] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      return(1);
    }

    fini_1level_ztable ( &aff_buffer );

    rtime = _GET_TIME - rtime;
    if (g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_write_scalar_aff] time for contract_diagram_write_scalar_aff = %e seconds %s %d\n", rtime, __FILE__, __LINE__);
  }  /* end of if io_proc == 2 */
  return(0);
}  // end of contract_diagram_write_scalar_aff
#endif  /* of HAVE_LHPC_AFF */

/***********************************************/
/***********************************************/
/***********************************************
 * write contracted diagram to FILE*
 ***********************************************/
int contract_diagram_write_fp ( double _Complex*** const diagram, FILE *fp, char*tag, int const tstart, unsigned int const dt, int const fbwd  ) {

  double rtime;

  rtime = _GET_TIME;

  if ( ( fbwd != 0 ) && ( dt != T_global - 1 ) ) {
    fprintf ( stderr, "[contract_diagram_write_fp] Error, incompatible dt and fbwd arguments %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fprintf ( fp, "# %s\n", tag );

  if ( fbwd == 0 ) {
    for ( unsigned int t = 0; t <= dt; t++ ) {
      for ( int mu = 0; mu < 4; mu++ ) {
      for ( int nu = 0; nu < 4; nu++ ) {
        fprintf ( fp, "  %25.16e  %25.16e\n", creal ( diagram[t][mu][nu] ), cimag ( diagram[t][mu][nu] ) );
      }}
    }
  } else {
    for ( unsigned int i = 0; i <= dt; i++ ) {
      int t = ( tstart + i * fbwd  + T_global ) % T_global;
      for ( int mu = 0; mu < 4; mu++ ) {
      for ( int nu = 0; nu < 4; nu++ ) {
        fprintf ( fp, "%25.16e  %25.16e\n", creal ( diagram[t][mu][nu] ), cimag ( diagram[t][mu][nu] ) );
      }}
    }
  }

  rtime = _GET_TIME - rtime;
  if (g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [contract_diagram_write_fp] time for contract_diagram_write_fp = %e seconds %s %d\n", rtime, __FILE__, __LINE__);

  return(0);
}  /* end of contract_diagram_write_fp */

/***********************************************/
/***********************************************/

/***********************************************
 * printf block of data
 ***********************************************/
void printf_data_from_key ( char *key_name, double _Complex **key_data, int const N1, int const N2, FILE*ofs ) {
  if ( key_name == NULL || key_data == NULL || N1 <= 0 || N2 <= 0 ) return;
  if ( ofs == NULL ) ofs = stdout;

  fprintf ( ofs, "# [printf_data_from_key] show key %s\n", key_name );
  for ( int i = 0; i < N1; i++ ) {
    for ( int k = 0; k < N2; k++ ) {
      fprintf ( ofs, "  %3d %4d   %25.15e %25.16e\n", i, k, creal( key_data[i][k]), cimag( key_data[i][k]) );
    }
  }
  return;
}  // end of printf_data_from_key



/***********************************************/
/***********************************************/

/***********************************************
 * 
 ***********************************************/
#ifdef HAVE_LHPC_AFF
int contract_diagram_read_key_qlua ( 
    double _Complex **fac, // output
    char const *prefix,    // key prefix
    int const gi,          // sequential gamma id
    int const pi[3],       // sequential momenta
    int const gsx[4],      // source coords
    int const isample,     // number of sample
    char const * vtype,    // contraction type
    int const gf,          // vertex gamma
    int const pf[3],       // vertex momentum
    struct AffReader_s * affr,  // AFF reader 
    int const N,           // length of data key ( will be mostly T_global )
    int const ncomp        // number of components
  ) {

  char key_prefix[400];
  char pf_str[20];
  char gi_str[20] = "";
  char gf_str[20] = "";
  char pi_str[30] = "";
  char isample_str[20] = "";
  int exitstatus;
  struct AffNode_s *affn = NULL, *affdir = NULL, *affpath = NULL;
  char *aff_errstr = NULL;

  if( (affn = aff_reader_root( affr )) == NULL ) {
    fprintf(stderr, "[contract_diagram_read_key_qlua] Error, aff reader is not initialized\n");
    return(103);
  }


  /* pf as momentum string */
  sprintf ( pf_str, "PX%d_PY%d_PZ%d", pf[0], pf[1], pf[2] );

  if ( pi != NULL ) {
    sprintf ( pi_str, "pi2x%.2dpi2y%.2dpi2z%.2d/", pi[0], pi[1], pi[2] );
  }
  if ( ( strcmp( vtype , "t1") == 0 ) ||
       ( strcmp( vtype , "t2") == 0 ) ||
       ( strcmp( vtype , "m1") == 0 ) ) {

      sprintf ( gf_str, "gf%.2d_gi%.2d/", gf, gi );
  } else {
    if ( gi > -1 ) {
      sprintf ( gi_str, "gi2%.2d/", gi );
    }
    if ( gf > -1 ) {
      sprintf ( gf_str, "gf%.2d/", gf );
    }
  }

  if ( isample  > -1 ) {
    sprintf ( isample_str, "sample%.2d/", isample );
  }

  sprintf ( key_prefix, "/%s/%s%st%.2dx%.2dy%.2dz%.2d/%s%s/%s",
      prefix, pi_str, gi_str, gsx[0], gsx[1], gsx[2], gsx[3], isample_str, vtype, gf_str );

  if ( g_verbose > 2 ) fprintf ( stdout, "# [contract_diagram_read_key_qlua] current key prefix %s\n", key_prefix );

  affdir = aff_reader_chpath (affr, affn, key_prefix );
  if ( ( aff_errstr = (char*)aff_reader_errstr ( affr ) ) != NULL ) {
    fprintf(stderr, "[contract_diagram_read_key_qlua] Error from aff_reader_chpath for key prefix \"%s\", status was %s\n", key_prefix, aff_errstr );
    return(115);
  } 

  if ( ncomp == 1 ) {

    affpath = aff_reader_chpath (affr, affdir, pf_str );
    exitstatus = aff_node_get_complex (affr, affpath, fac[0], N );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[contract_diagram_read_key_qlua] Error from aff_node_get_complex for key \"%s\" + \"%s\", status was %d\n", key_prefix, pf_str, exitstatus);
      return(105);
    }

  } else {

    double _Complex buffer[N];
    char key[30];

    for ( int icomp = 0; icomp < ncomp; icomp++ ) {

      sprintf ( key, "c%d/%s", icomp, pf_str );

      affpath = aff_reader_chpath (affr, affdir, key );
      exitstatus = aff_node_get_complex (affr, affpath, buffer, N );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[contract_diagram_read_key_qlua] Error from aff_node_get_complex for key \"%s\" + \"%s\", status was %d\n", key_prefix, key, exitstatus);
        return(105);
      }

      for ( int it = 0; it < N; it++ ) fac[it][icomp] = buffer[it];

    }  // end of loop on components

  }

  /***********************************************
   * show data
   ***********************************************/
  if ( g_verbose > 4 ) {
    sprintf ( key_prefix, "%s/%s", key_prefix, pf_str );
    printf_data_from_key ( key_prefix, fac, N, ncomp, stdout );
  }

  return(0);
}  // end of contract_diagram_read_key_qlua
#endif  /* end of if def HAVE_LHPC_AFF */

/***********************************************/
/***********************************************/

/***********************************************
 * 
 ***********************************************/
#ifdef HAVE_LHPC_AFF
int contract_diagram_read_oet_key_qlua ( 
    double _Complex ***fac, // output
    char const *prefix,     // key prefix
    int const pi[3],        // sequential momenta
    int const gsx[4],       // source coords
    char const * vtype,     // contraction type
    int const gf,           // vertex gamma
    int const pf[3],        // vertex momentum
    struct AffReader_s *affr,  // AFF reader 
    int const N,            // length of data key ( will be mostly T_global )
    int const ncomp         // components
  ) {

  char key_prefix[400];
  char key[500];
  char pf_str[20];
  char pi_str[30] = "";
  int exitstatus;
  struct AffNode_s *affn = NULL, *affdir = NULL, *affpath = NULL;
  double _Complex buffer[N];
  char * aff_errstr = NULL;

  if( (affn = aff_reader_root( affr )) == NULL ) {
    fprintf(stderr, "[contract_diagram_read_oet_key_qlua] Error, aff reader is not initialized\n");
    return(103);
  }


  /* pf as momentum string */
  sprintf ( pf_str, "PX%d_PY%d_PZ%d", pf[0], pf[1], pf[2] );

  if ( pi != NULL ) {
    sprintf ( pi_str, "pi2x%.2dpi2y%.2dpi2z%.2d/", pi[0], pi[1], pi[2] );
  } else {
    sprintf ( pi_str, "pi2x%.2dpi2y%.2dpi2z%.2d/", 0, 0, 0 );
  }

  for ( int k = 0; k < 4; k++ ) {

    sprintf ( key_prefix, "/%s/%st%.2dx%.2dy%.2dz%.2d/dphi%d/%s/gf%.2d", prefix, pi_str,  gsx[0], gsx[1], gsx[2], gsx[3], k, vtype, gf );

    affdir = aff_reader_chpath (affr, affn, key_prefix );
    if ( ( aff_errstr = (char*)aff_reader_errstr ( affr ) ) != NULL ) {
      fprintf(stderr, "[contract_diagram_read_oet_key_qlua] Error from aff_reader_chpath for key prefix \"%s\", status was %s %s %d\n", key_prefix, aff_errstr, __FILE__, __LINE__ );
      return(125);
    } 

    if ( g_verbose > 2 ) fprintf ( stdout, "# [contract_diagram_read_oet_key_qlua] current key prefix %s\n", key_prefix );

    if ( ncomp == 1 ) {

      affpath = aff_reader_chpath (affr, affdir, pf_str );
      exitstatus = aff_node_get_complex (affr, affpath, fac[k][0], N );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[contract_diagram_read_oet_key_qlua] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", key, exitstatus, __FILE__, __LINE__);
        return(126);
      }

    } else {
      for ( int icomp = 0; icomp < ncomp; icomp++ ) {

        sprintf ( key, "c%d/%s", icomp, pf_str );

        affpath = aff_reader_chpath (affr, affdir, key );
        exitstatus = aff_node_get_complex (affr, affpath, buffer, N );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[contract_diagram_read_oet_key_qlua] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", key, exitstatus, __FILE__, __LINE__ );
          return(127);
        }

        for ( int it = 0; it < N; it++ ) fac[k][it][icomp] = buffer[it];
      }

    }  // end of if ncomp == 1 else

    /***********************************************
     * show data
     ***********************************************/
    if ( g_verbose > 4 ) {
      sprintf ( key_prefix, "%s/%s", key_prefix, pf_str );
      printf_data_from_key ( key_prefix, fac[k], N, ncomp, stdout );
    }

  }  // end of loop on spinor component k

  return(0);
}  // end of contract_diagram_read_oet_key_qlua

#endif  /* end of if def HAVE_LHPC_AFF */

/***********************************************/
/***********************************************/

/***********************************************
 *
 ***********************************************/
int contract_diagram_key_suffix ( char * const suffix, int const gf2, int const pf2[3], int const gf11, int const gf12, int const pf1[3], int const gi2, int const pi2[3], int const gi11, int const gi12, int const pi1[3], int const sx[4]  ) {

  char sx_str[40] = "";
  char gf2_str[40] = "";
  char gf1_str[40] = "";
  char gi2_str[40] = "";
  char gi1_str[40] = "";
  char pf2_str[40] = "";
  char pf1_str[40] = "";
  char pi2_str[40] = "";
  char pi1_str[40] = "";

  if ( sx != NULL ) { sprintf ( sx_str, "/t%.2dx%.2dy%.2dz%.2d", sx[0], sx[1], sx[2], sx[3] ); }

  if ( gf2 > -1 ) { sprintf( gf2_str, "/gf2%.2d", gf2 ); }

  if ( gf11 > -1 ) { 
    sprintf( gf1_str, "/gf1%.2d", gf11 ); 
    if ( gf12 > -1 ) { 
      sprintf( gf1_str, "%s_%.2d", gf1_str, gf12 ); 
    }
  }

  if ( gi2 > -1 ) { sprintf( gi2_str, "/gi2%.2d", gi2 ); }

  if ( gi11 > -1 ) { 
    sprintf( gi1_str, "/gi1%.2d", gi11 ); 
    if ( gi12 > -1 ) { 
      sprintf( gi1_str, "%s_%.2d", gi1_str, gi12 ); 
    }
  }

  if ( pf2 != NULL ) { sprintf ( pf2_str, "/pf2x%.2dpf2y%.2dpf2z%.2d", pf2[0], pf2[1], pf2[2] ); }

  if ( pf1 != NULL ) { sprintf ( pf1_str, "/pf1x%.2dpf1y%.2dpf1z%.2d", pf1[0], pf1[1], pf1[2] ); }

  if ( pi2 != NULL ) { sprintf ( pi2_str, "/pi2x%.2dpi2y%.2dpi2z%.2d", pi2[0], pi2[1], pi2[2] ); }

  if ( pi1 != NULL ) { sprintf ( pi1_str, "/pi1x%.2dpi1y%.2dpi1z%.2d", pi1[0], pi1[1], pi1[2] ); }

  // sprintf( suffix, "gf2%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/gf1%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d%s", 
  //    gf2, pf2[0], pf2[1], pf2[2], gf1, pf1[0], pf1[1], pf1[2], gi2, pi2[0], pi2[1], pi2[2], gi1, sx_str );

  sprintf( suffix, "%s%s%s%s%s%s%s%s%s", gf2_str, pf2_str, gf1_str, pf1_str, gi2_str, pi2_str, gi1_str, pi1_str, sx_str );
 
  if ( g_cart_id == 0 && g_verbose > 2 ) fprintf ( stdout, "# [contract_diagram_key_suffix] key suffix = %s\n", suffix );

  return(0);

}  // end of contract_diagram_key_suffix  

/***********************************************/
/***********************************************/

/***********************************************
 * 
 ***********************************************/
int contract_diagram_key_suffix_from_type ( char * key_suffix, twopoint_function_type * p ) {

  int exitstatus;

  if ( strcmp( p->type, "mxb-mxb" ) == 0 ) {

    exitstatus = contract_diagram_key_suffix ( key_suffix, p->gf2, p->pf2, p->gf1[0], p->gf1[1], p->pf1, p->gi2, p->pi2, p->gi1[0], p->gi1[1], p->pi1, p->source_coords );

  } else if ( strcmp( p->type, "mxb-b" ) == 0 ) {

    exitstatus = contract_diagram_key_suffix ( key_suffix,     -1,   NULL, p->gf1[0], p->gf1[1], p->pf1, p->gi2, p->pi2, p->gi1[0], p->gi1[1], p->pi1, p->source_coords );

  } else if ( strcmp( p->type, "b-b" ) == 0 ) {

    exitstatus = contract_diagram_key_suffix ( key_suffix,     -1,   NULL, p->gf1[0], p->gf1[1], p->pf1,      -1,  NULL, p->gi1[0], p->gi1[1], p->pi1, p->source_coords );

  } else if ( strcmp( p->type, "m-m" ) == 0 ) {

    exitstatus = contract_diagram_key_suffix ( key_suffix, p->gf2, p->pf2,        -1,         -1,  NULL, p->gi2, p->pi2,        -1,        -1,   NULL, p->source_coords );

  } else {
    fprintf ( stderr, "[contract_diagram_key_suffix_from_type] unknown twopoint_function type %s %s %d\n", p->type, __FILE__, __LINE__ );
    return( 1 );
  }
  return ( 0 );
}  // end of contract_diagram_key_suffix_from_type

/***********************************************/
/***********************************************/

/***********************************************
 * multiply zm4x4 list with gamma matrix
 ***********************************************/
int contract_diagram_zm4x4_field_mul_gamma_lr ( double _Complex *** const sp_out, double _Complex *** const sp_in, gamma_matrix_type const gl, gamma_matrix_type const gr, unsigned int const N ) {

  double _Complex *** sp_aux = init_3level_ztable ( N, 4, 4 );
  if ( sp_aux == NULL ) { return(1); }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    // sp_aux <- sp_in x Gamma_r
    zm4x4_eq_zm4x4_ti_zm4x4 ( sp_aux[it], sp_in[it],  (double _Complex**)gr.m );

    // diagram <- Gamma_f1_1 x diagram_buffer
    zm4x4_eq_zm4x4_ti_zm4x4 ( sp_out[it], (double _Complex**)gl.m, sp_aux[it] );
  }
  
  fini_3level_ztable ( &sp_aux );

  return (0);
}  // end of contract_diagram_zm4x4_field_mul_gamma_lr

/***********************************************/
/***********************************************/

/********************************************************************************
 * baryon at source gets (1) from adjoint current construction and -I from 
 * factor I in C, which comes adjoint
 ********************************************************************************/
double _Complex contract_diagram_get_correlator_phase ( char * const type, int const gi11, int const gi12, int const gi2, int const gf11, int const gf12, int const gf2 ) {

  double _Complex zsign = 0.;

  if ( strcmp( type , "b-b" ) == 0 ) {
    /********************************************************************************
     * baryon - baryon 2-point function
     *
     ********************************************************************************/

    zsign = (double _Complex) ( get_gamma_signs ( "g0d" , gi11 ) * get_gamma_signs ( "g0d" , gi12 ) ) * (-1) * (I) * (-I);
        // sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] 

  } else if ( strcmp( type , "mxb-b" ) == 0 ) {
    /********************************************************************************
     * meson-baryon - baryon 2-point function
     *
     * meson-baryon at source
     *
     ********************************************************************************/

    int const sigma_gi2 = get_gamma_signs ( "g0d", gi2 );

    zsign = (double _Complex) ( get_gamma_signs ( "g0d" , gi11 ) * get_gamma_signs ( "g0d" , gi12 ) ) * (-1) * (I) * (-I);
        // sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] * sigma_gamma_adj_g0_dagger[p->gi2] 
        
    zsign *= ( sigma_gi2 == -1 ) ? I : 1.;

  } else if ( strcmp( type , "mxb-mxb" ) == 0 ) {
    /********************************************************************************
     * meson-baryon - meson-baryon
     *
     * at source and sink
     ********************************************************************************/

    int const sigma_gi2 = get_gamma_signs ( "g0d", gi2 );
    int const sigma_gf2 = get_gamma_signs ( "g0d", gf2 );

    zsign = (double _Complex) ( get_gamma_signs ( "g0d" , gi11 ) * get_gamma_signs ( "g0d" , gi12 ) ) * (-1) * (I) * (-I);
        // sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] * sigma_gamma_adj_g0_dagger[p->gi2] 
        
    zsign *= ( sigma_gi2 == -1 ) ? I : 1.;
    zsign *= ( sigma_gf2 == -1 ) ? I : 1.;

  } else if ( strcmp( type , "m-m" ) == 0 ) {
    /********************************************************************************
     * meson - meson
     ********************************************************************************/

    int const sigma_gi2 = get_gamma_signs ( "g0d", gi2 );
    int const sigma_gf2 = get_gamma_signs ( "g0d", gf2 );

    zsign = ( ( sigma_gi2 == -1 ) ? I : 1 ) * ( ( sigma_gf2 == -1 ) ? I : 1 );

  } else if ( strcmp( type , "mxb-J-b" ) == 0 ) {
    /********************************************************************************
     * meson-baryon - current insertion - baryon
     *
     * at source and sink
     ********************************************************************************/

    int const sigma_gi2 = get_gamma_signs ( "g0d", gi2 );

    zsign = (double _Complex) ( get_gamma_signs ( "g0d" , gi11 ) * get_gamma_signs ( "g0d" , gi12 ) ) * (-1) * (I) * (-I);

    zsign *= ( sigma_gi2 == -1 ) ? I : 1.;

  } else if ( strcmp( type , "b-J-b" ) == 0 ) {
    /********************************************************************************
     * baryon - current insertion - baryon
     *
     * at source and sink
     ********************************************************************************/

    zsign = (double _Complex) ( get_gamma_signs ( "g0d" , gi11 ) * get_gamma_signs ( "g0d" , gi12 ) ) * (-1) * (I) * (-I);

  }  /* end of if type */

  if (g_verbose > 2) {
    fprintf(stdout, "# [contract_diagram_get_correlator_phase] gf1 %2d - %2d\n"\
                    "                                          gf2 %2d\n"\
                    "                                          gi1 %2d - %2d\n"\
                    "                                          gi2 %2d\n"\
                    "                                          phase %3.0f  %3.0f\n",
        gf11, gf12, gf2, gi11, gi12, gi2, creal(zsign), cimag(zsign) );
  }

  return( zsign );

}  // end of contract_diagram_get_correlator_phase

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *  
 *  safe, if sp_out = sp_in in memory
 ********************************************************************************/
void contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op ( double _Complex *** const sp_out, double _Complex ** const R1, char const op1 , double _Complex *** const sp_in, double _Complex ** const R2, char const op2, unsigned int const N ) {

  int const dim = 4;

  double _Complex ** S1 = init_2level_ztable ( dim, dim );
  double _Complex ** S2 = init_2level_ztable ( dim, dim );

  /* apply op1 to R1 */
  if ( op1 == 'N' ) {
    rot_mat_assign ( S1, R1, dim );
  } else if ( op1 == 'C' ) {
    rot_mat_conj   ( S1, R1, dim );
  } else if ( op1 == 'H' ) {
    rot_mat_adj    ( S1, R1, dim );
  } else if ( op1 == 'T' ) {
    rot_mat_trans  ( S1, R1, dim );
  }
  /* apply op2 to R2 */
  if ( op2 == 'N' ) {
    rot_mat_assign ( S2, R2, dim );
  } else if ( op2 == 'C' ) {
    rot_mat_conj   ( S2, R2, dim );
  } else if ( op2 == 'H' ) {
    rot_mat_adj    ( S2, R2, dim );
  } else if ( op2 == 'T' ) {
    rot_mat_trans  ( S2, R2, dim );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel shared ( S1, S2 )
  {
#endif
  double _Complex ** sp_aux = init_2level_ztable ( dim, dim );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {
    /* sp_aux = S1 sp_in = R1^op1 sp_in */
    rot_mat_ti_mat ( sp_aux,     S1,     sp_in[it], dim );
    /* sp_out = sp_aux S2 = R1^op1 sp_in R2^op2 */
    rot_mat_ti_mat ( sp_out[it], sp_aux, S2,        dim );
  }

  fini_2level_ztable ( &sp_aux );
#ifdef HAVE_OPENMP
  }  /* end of parallel region */
#endif

  fini_2level_ztable ( &S1 );
  fini_2level_ztable ( &S2 );

}  /* end of contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op */


/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *  
 ********************************************************************************/
int contract_diagram_finalize ( double _Complex *** const diagram, char * const diagram_type, int const sx[4], int const p[3], 
    int const gf11_id, int const gf12_id, int const gf12_sign, int const gf2_id,
    int const gi11_id, int const gi12_id, int const gi12_sign, int const gi2_id,
    unsigned int const N )
{

  int exitstatus;

  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [contract_diagram_finalize] sx = ( %3d, %3d, %3d, %3d ) p = (%3d, %3d, %3d) gf = (%3d, %3d; %3d) sf12 %2d gi = (%3d, %3d; %3d) si12 %2d \n",
        sx[0], sx[1], sx[2], sx[3], p[0], p[1], p[2],
        gf11_id, gf12_id, gf2_id, gf12_sign, gi11_id, gi12_id, gi2_id, gi12_sign );
  }


  /*******************************************
   * add boundary phase
   *******************************************/
  correlator_add_baryon_boundary_phase ( diagram, sx[0], +1, N );

  /*******************************************
   * add momentum phase at source
   *******************************************/
  correlator_add_source_phase ( diagram, p, &(sx[1]), N );

  /*******************************************
   * add overall phase factor
   *******************************************/
  double _Complex const zsign = contract_diagram_get_correlator_phase ( diagram_type, gi11_id, gi12_id, gi2_id, gf11_id, gf12_id, gf2_id );

  contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, N );

  /*******************************************
   * add outer gamma matrices
   *******************************************/
  gamma_matrix_type gf12;
  gamma_matrix_set ( &gf12, gf12_id, gf12_sign );

  gamma_matrix_type gi12;
  gamma_matrix_set ( &gi12, gi12_id, gi12_sign );

  if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, N ) ) != 0 ) {
    fprintf( stderr, "[contract_diagram_finalize] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }


  return(0);
}  /* end of contract_diagram_finalize */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *  
 ********************************************************************************/
int contract_diagram_co_eq_tr_zm4x4_field ( double _Complex * const r, double _Complex *** const diagram, unsigned int const N ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int t = 0; t < N; t++ ) {

    co_eq_tr_zm4x4 ( r+t, diagram[t] );
  }
  return ( 0 );
}

/********************************************************************************/
/********************************************************************************/

}  // end of namespace cvc
