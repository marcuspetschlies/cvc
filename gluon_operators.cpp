/********************
 * gluon_operators
 *
 ********************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "cvc_utils.h"
#include "matrix_init.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "Q_clover_phi.h"
#include "gluon_operators.h"

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

#undef _SYM_ACTION

/********************************************************************/
/********************************************************************/

namespace cvc {

#if 0
inline void project_to_generators ( double * const p, double * const A ) {

  const double sq1  = 0.8164965809277260;  /* sqrt ( 2 / 3 ) */
  const double sq2  = 0.5773502691896257;  /* sqrt ( 1 / 3 ) */

  /* unit matrix */
  p[0] = ( A[1] + A[9] + A[17] ) * sq1;

  /* lambda_1 */
  p[1] = A[3] + A[ 7];

  /* lambda_2 */
  p[2] = A[2] - A[ 6];

  /* lambda_3 */
  p[3] = A[1] - A[ 9];

  /* lambda_4 */
  p[4] = A[5] + A[13];

  /* lambda_5 */
  p[5] = A[4] - A[12];

  /* lambda_6 */
  p[6] = A[11] + A[15];

  /* lambda_7 */
  p[7] = A[10] - A[14];

  /* lambda_2 / 2 */
  p[8] = ( A[1] + A[9] - 2 * A[17] ) * sq2;
}  /* end of project_to_generators */

inline void restore_from_generators ( double * const A, double * const p ) {
  const double one_over_sqrt3  = 0.5773502691896258;
  const double one_over_sqrt6  = 0.4082482904638631;

  A[ 0] = p[0] * one_over_sqrt6 + p[3] * 0.5 + p[8] * one_over_sqrt3;
  A[ 1] = 0.;
  A[ 2] =  p[1] * 0.5;
  A[ 3] = -p[2] * 0.5;
  A[ 4] =  p[4] * 0.5;
  A[ 5] = -p[5] * 0.5;
  A[ 6] =  p[1] * 0.5;
  A[ 7] =  p[2] * 0.5;
  A[ 8] = p[0] * one_over_sqrt6 - p[3] * 0.5 + p[8] * one_over_sqrt3;
  A[ 9] = 0.;
  A[10] =  p[6] * 0.5;
  A[11] = -p[7] * 0.5;
  A[12] =  p[4] * 0.5;
  A[13] =  p[5] * 0.5;
  A[14] =  p[6] * 0.5;
  A[15] =  p[7] * 0.5;
  A[16] = p[0] * one_over_sqrt6 - p[8] * 2 * one_over_sqrt3;
  A[17] = 0.;
}
#endif  /* of if 0 */

/********************************************************************/
/********************************************************************/

/********************************************************************
 * out
 *   Gp : non-zero field strength tensor components
 *        from plaquettes, 1x1 loops
 *   Gr : non-zero field strength tensor components
 *        from rectangles, 1x2 and 2x1 loops  
 ********************************************************************/
int G_plaq_rect ( double *** Gp, double *** Gr, double * const gauge_field) {

  const int dirpairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
 
  const double one_over_four  = 0.250;
  const double one_over_eight = 0.125;

#ifdef HAVE_MPI
  xchanger_type x_plaq, x_rect;
  mpi_init_xchanger ( &x_plaq, 108 );
  mpi_init_xchanger ( &x_rect, 432 );
#endif


  double *** plaquettes = init_3level_dtable ( VOLUMEPLUSRAND, 6, 18 );
  if ( plaquettes == NULL ) {
    fprintf ( stderr, "[G_plaq_rect] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

#ifdef HAVE_MPI
  /********************************************************************
   * exchange gauge field to be sure
   ********************************************************************/
  xchange_gauge_field ( gauge_field );
#endif

  /********************************************************************
   * calculate elementary plaquettes for all triples (ix, mu, nu )
   ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
  
    double U1[18], U2[18];

    for ( int imunu = 0; imunu < 6; imunu++) {
       
      const int imu = dirpairs[imunu][0];
      const int inu = dirpairs[imunu][1];

        /********************************************************************
         * P <- [ U_mu(x) U_nu(x+mu) ] * [ U_nu(x) U_mu(x+nu) ]^+
         *   =  U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
         *
         * x+nu >    x+mu+nu
         *   _______
         *  |       |
         *  |       |
         * ^|       | 
         * _|_______|
         *  |x  <    x + mu
         *
         ********************************************************************/
        /********************************************************************
         * 2 corners,
         * (U1) U_mu(x) * U_nu(x+mu)
         * (U2) U_nu(x) * U_mu(x+nu)
         * then multply 
         * U1 x U2^+ = U_mu(x) * U_nu(x+mu) * U_mu(x+nu)^+ * U_nu(x)^+
         *           = U(x,x+mu) * U(x+mu,x+mu+nu) * U(x+nu+mu,x+nu) * U(x+nu,x)
         ********************************************************************/
        _cm_eq_cm_ti_cm ( U1, gauge_field + _GGI(ix, imu), gauge_field + _GGI( g_iup[ix][imu], inu) );
        _cm_eq_cm_ti_cm ( U2, gauge_field + _GGI(ix, inu), gauge_field + _GGI( g_iup[ix][inu], imu) );
        _cm_eq_cm_ti_cm_dag ( plaquettes[ix][imunu], U1, U2 );
 
    }  /* end of loop on direction pairs */
  }  /* end of loop on VOLUME */

  /********************************************************************
   * xchange the plaquettes,
   *   INCLUDING edges
   ********************************************************************/
#ifdef HAVE_MPI
  mpi_xchanger ( plaquettes[0][0], &x_plaq );
#endif

#if 0
  /********************************************************************
   * TEST: calculate averaged plaquettes
   ********************************************************************/
  double ** pl = init_2level_dtable ( 2, 2*T );
  const unsigned int VOL3 = LX*LY*LZ;
  for ( int it = 0; it < T; it++ ) {
    for ( unsigned int ix = 0; ix < VOL3; ix++ ) {
      d2_pl_eq_tr_cm ( pl[0]+2*it, plaquettes[it*VOL3+ix][0]  );
      d2_pl_eq_tr_cm ( pl[0]+2*it, plaquettes[it*VOL3+ix][1]  );
      d2_pl_eq_tr_cm ( pl[0]+2*it, plaquettes[it*VOL3+ix][2]  );

      d2_pl_eq_tr_cm ( pl[1]+2*it, plaquettes[it*VOL3+ix][3]  );
      d2_pl_eq_tr_cm ( pl[1]+2*it, plaquettes[it*VOL3+ix][4]  );
      d2_pl_eq_tr_cm ( pl[1]+2*it, plaquettes[it*VOL3+ix][5]  );
    }
    fprintf ( stdout, "# [G_plaq_rect] t %3d     plt %25.16e %25.16e     pl %25.16e %25.16e\n",
        it + g_proc_coords[0]*T, pl[0][2*it], pl[0][2*it+1], pl[1][2*it], pl[1][2*it+1] );
  }
  fini_2level_dtable ( &pl );
  /********************************************************************
   * end of TEST
   ********************************************************************/
#endif  /* for TEST */

  /********************************************************************
   * build G_munu from plaquettes
   ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(plaquettes)
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
    double U1[18], U2[18], U3[18], U4[18];
    
    for ( int imunu = 0; imunu < 6; imunu++) {

      const int imu = dirpairs[imunu][0]; 
      const int inu = dirpairs[imunu][1]; 

      _cm_eq_zero ( U1 );
      
      /********************************************************************
       * (1) U1 += U_{mu,nu}(x)
       ********************************************************************/
      _cm_pl_eq_cm ( U1, plaquettes[ix][imunu] );
#if 0
#endif

      /********************************************************************
       * (2) U1 +=  U_{mu,nu}(x - mu)
       ********************************************************************/
      _cm_eq_cm_ti_cm( U2, plaquettes[ g_idn[ix][imu] ][imunu], gauge_field + _GGI( g_idn[ix][imu], imu ) );
      _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][imu], imu ), U2 );
      _cm_pl_eq_cm ( U1, U3 );
#if 0
#endif

      /********************************************************************
       * (3) U1 += U_{mu,nu}(x - nu)
       ********************************************************************/
      _cm_eq_cm_ti_cm( U2, plaquettes[ g_idn[ix][inu] ][imunu], gauge_field + _GGI( g_idn[ix][inu], inu ) );
      _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][inu], inu ), U2 );
      _cm_pl_eq_cm ( U1, U3 );
#if 0
#endif

      /********************************************************************
       * (4) U1 += U_{mu,nu}(x - nu - mu)
       ********************************************************************/
      _cm_eq_cm_ti_cm ( U4, gauge_field + _GGI(g_idn[g_idn[ix][imu]][inu], imu), gauge_field + _GGI(g_idn[ix][inu], inu) );
      _cm_eq_cm_ti_cm ( U2, plaquettes[ g_idn[ g_idn[ix][inu] ][imu] ][imunu],  U4 );
      _cm_eq_cm_dag_ti_cm ( U3, U4, U2 );
      _cm_pl_eq_cm ( U1, U3 );
#if 0
#endif
      _cm_eq_antiherm_cm ( Gp[ix][imunu], U1 );
/* TEST      _cm_eq_cm ( Gp[ix][imunu], U1 ); */
      _cm_ti_eq_re ( Gp[ix][imunu], one_over_four );

      /********************************************************************
       * at high verbosity write G_plaq
       ********************************************************************/
      if ( g_verbose > 4 ) {
        for( int ia = 0; ia < 9; ia++ ) {
          fprintf ( stdout, "Gp %3d %3d %3d %3d    %d %d    %d %d    %25.16e %25.16e\n",
               ix                           / (LX*LY*LZ) + g_proc_coords[0]*T,
              (ix            % (LX*LY*LZ) ) / (LY*LZ)    + g_proc_coords[1]*LX,
              (ix            % (LY*LZ)    ) / (LZ)       + g_proc_coords[2]*LY,
              (ix            % LZ         )              + g_proc_coords[3]*LZ,
              imu, inu, ia/3, ia%3,
              Gp[ix][imunu][2*ia], Gp[ix][imunu][2*ia+1] );
        }
        fprintf ( stdout, "# Gp\n" );
      }

    }  /* end of loop on hyperplanes */
  }  /* end of loop on volume */

#if 0
  /********************************************************************
   * TEST, compare to g_clover
   ********************************************************************/
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
    int ieo = 1 - g_iseven[ix];

    for ( int ic = 0; ic < 6; ic++ ) {
      for( int k =0; k < 9; k++ ) {
        fprintf ( stdout, "x %6d mu %d nu %d c %d %d  G %25.16e %25.16e  CL %25.16e %25.16e\n", ix, dirpairs[ic][0] , dirpairs[ic][1],
            k/3, k%3, Gp[ix][ic][2*k], Gp[ix][ic][2*k+1],
            g_clover[ieo][_GSWI( g_lexic2eosub[ix],ic) + 2*k], g_clover[ieo][_GSWI( g_lexic2eosub[ix],ic) + 2*k+1] );
      }
    }
  }
  /********************************************************************
   * end of TEST
   ********************************************************************/
#endif  /* of if 0 */

  /********************************************************************
   * now build the rectangles from products of plaquettes
   *
   * needs rotation of plaquettes taken from neighbours to current site
   ********************************************************************/
  double ***** rectangles = init_5level_dtable ( VOLUMEPLUSRAND, 6, 2, 2, 18 );
  if ( rectangles == NULL ) {
    fprintf ( stderr, "[G_plaq_rect] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(plaquettes)
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

    for ( int imunu = 0; imunu < 6; imunu++) {

      const int imu = dirpairs[imunu][0]; 
      const int inu = dirpairs[imunu][1]; 

      double U1[18], U2[18], U3[18];

        /**********************************************
         *
         * R_mu,nu,0,0 <- P_mu,nu,x-mu * P_mu,nu,x
         *
         **********************************************/
        _cm_eq_cm_ti_cm ( U1, plaquettes[g_idn[ix][imu]][imunu] , gauge_field + _GGI( g_idn[ix][imu], imu ) );
        _cm_eq_cm_dag_ti_cm ( U2, gauge_field + _GGI( g_idn[ix][imu], imu ), U1 );
        _cm_eq_cm_ti_cm ( rectangles[ix][imunu][0][0], plaquettes[ix][imunu], U2 );
    

        /**********************************************
         *
         * R_mu,nu,0,1 <- P_mu,nu,x-mu-nu * P_mu,nu,x-nu
         *
         **********************************************/
        
        _cm_eq_cm_ti_cm ( U1, plaquettes[g_idn[g_idn[ix][imu]][inu]][imunu], gauge_field + _GGI( g_idn[ g_idn[ix][imu] ][inu], imu ) );
        _cm_eq_cm_dag_ti_cm ( U2, gauge_field + _GGI( g_idn[ g_idn[ix][imu] ][inu], imu ) , U1 );
 
        _cm_eq_cm_ti_cm ( U1, U2, plaquettes[g_idn[ix][inu]][imunu] );

        _cm_eq_cm_ti_cm ( U3, U1, gauge_field + _GGI( g_idn[ix][inu], inu ) );
        _cm_eq_cm_dag_ti_cm ( rectangles[ix][imunu][0][1], gauge_field + _GGI( g_idn[ix][inu], inu ) , U3 );


        /**********************************************
         *
         * R_mu,nu,1,1 <- P_mu,nu,x-mu-nu * P_mu,nu,x-mu
         *
         * REUSE U1 from above
         **********************************************/
        
        _cm_eq_cm_ti_cm( U1, U2, gauge_field + _GGI( g_idn[ix][inu], inu) );
        _cm_eq_cm_dag_ti_cm( U2,  gauge_field + _GGI( g_idn[ix][inu], inu), U1 );

        _cm_eq_cm_ti_cm( U1, plaquettes[g_idn[ix][imu]][imunu], gauge_field + _GGI( g_idn[ix][imu], imu) );
        _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][imu], imu), U1 );

        _cm_eq_cm_ti_cm ( rectangles[ix][imunu][1][1], U3, U2 );

        /**********************************************
         *
         * R_mu,nu,1,0 <- P_mu,nu,x-nu * P_mu,nu,x
         *
         **********************************************/
        _cm_eq_cm_ti_cm( U1, plaquettes[g_idn[ix][inu]][imunu], gauge_field + _GGI( g_idn[ix][inu], inu ) );
        _cm_eq_cm_dag_ti_cm( U2, gauge_field + _GGI( g_idn[ix][inu], inu ), U1 );
        _cm_eq_cm_ti_cm ( rectangles[ix][imunu][1][0], U2, plaquettes[ix][imunu] );

    }  /* end of loop on imunu  */

  }  /* end of loop on ix */

#if 0
  /********************************************************************
   * TEST
   * or alternative calculation of rectangles
   ********************************************************************/
/* #pragma omp parallel for */
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {

    double U1[18], U2[18];
    double R[6][2][2][18];

    int imunu = 0;


    for ( int imunu = 0; imunu < 6; imunu++ ) {
    
      const int imu = dirpairs[imunu][0];
      const int inu = dirpairs[imunu][1];

       memset ( R[imunu][0][0], 0, 18 * sizeof( double ) );
       memset ( R[imunu][0][1], 0, 18 * sizeof( double ) );
       memset ( R[imunu][1][0], 0, 18 * sizeof( double ) );
       memset ( R[imunu][1][1], 0, 18 * sizeof( double ) );

      /* x -> x+mu -> x+mu+nu */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(g_iup[ix][imu],inu) );
      /* x -> x+mu -> x+mu+nu -> x+nu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], imu) );
      /* x -> x+mu+nu -> x+nu -> x+nu-mu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_idn[g_iup[ix][inu]][imu], imu) );
      /* x -> x+mu+nu -> x+nu -> x+nu-mu -> x-mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ix][imu], inu) );
      /* x -> x+mu+nu -> x+nu -> x+nu-mu -> x-mu -> x */
      _cm_eq_cm_ti_cm( R[imunu][0][0], U2, gauge_field+_GGI( g_idn[ix][imu], imu) );


      /* x-mu-nu -> x-nu -> x */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][inu], inu ), gauge_field+_GGI( g_idn[ ix][imu], imu) );
      /* x -> x-mu-nu -> x-nu */
      _cm_eq_cm_dag_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][inu] ][imu], imu) );
      /* x -> x-mu-nu -> x-nu -> x-nu+mu */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ix][inu], imu) );
      /* x -> x-mu-nu -> x-nu -> x-nu+mu -> x-mu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_idn[ix][inu] ][imu], inu) );
      /* x -> x-mu-nu -> x-nu -> x-nu+mu -> x-mu -> x */
      _cm_eq_cm_ti_cm_dag( R[imunu][0][1], U2, gauge_field+_GGI( ix, imu) );

      /* x -> x-nu -> x-nu+mu */
      _cm_eq_cm_dag_ti_cm( U1, gauge_field+_GGI( g_idn[ix][inu],inu ), gauge_field+_GGI(g_idn[ix][inu],imu) );
      /* x -> x-nu -> x-nu+mu -> x+mu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_idn[ix][inu] ][imu], inu) );
      /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_iup[ix][imu], inu) );
      /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu -> x+nu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], imu) );
      /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu -> x+nu -> x */
      _cm_eq_cm_ti_cm_dag( R[imunu][1][0], U2, gauge_field+_GGI( ix, inu) );

      /* x -> x+nu -> x+nu-mu */
      _cm_eq_cm_ti_cm_dag( U1, gauge_field+_GGI(ix,inu), gauge_field+_GGI( g_idn[ g_iup[ix][inu] ][imu], imu) );
      /* x -> x+nu -> x+nu-mu -> x-mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ix][imu], inu) );
      /* x -> x+nu -> x+nu-mu -> x-mu -> x-mu-nu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_idn[g_idn[ix][inu]][imu], inu) );
      /* x -> x+nu -> x+nu-mu -> x-mu -> x-mu-nu -> x-nu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][inu], imu) );
      /* x -> x+nu -> x+nu-mu -> x-mu -> x-mu-nu -> x-nu -> x */
      _cm_eq_cm_ti_cm( R[imunu][1][1], U2, gauge_field+_GGI( g_idn[ix][inu], inu) );

    }
    for ( int imunu = 0; imunu < 6; imunu++ ) {
      for ( int ic = 0; ic < 4; ic++ ) {
        for ( int k = 0; k < 9; k++ ) {
          fprintf ( stdout, "M %6d %d    %d %d    %d %d    %25.16e %25.16e     %25.16e%25.16e\n", ix, imunu, ic/2, ic%2, k/3, k%3,
              rectangles[ix][imunu][ic/2][ic%2][2*k],
              rectangles[ix][imunu][ic/2][ic%2][2*k+1],
              R[imunu][ic/2][ic%2][2*k],
              R[imunu][ic/2][ic%2][2*k+1] );
        }
        fprintf( stdout, "\n" );
      }
      fprintf ( stdout, "\n\n\n" );
    }
    fprintf ( stdout, "\n\n\n\n\n" );
  }  /* end of loop on ix */
  /********************************************************************
   * end of TEST
   ********************************************************************/
#endif  /* of TEST */

  /********************************************************************
   * xchange the rectangles
   *   INCLUDING edges ???
   ********************************************************************/
#ifdef HAVE_MPI
  mpi_xchanger ( rectangles[0][0][0][0], &x_rect );
#endif


  /********************************************************************
   * build G_munu from rectangles
   ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(rectangles)
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
  
    double U1[18], U2[18];
    
    for ( int imunu = 0; imunu < 6; imunu++) {

      const int imu = dirpairs[imunu][0]; 
      const int inu = dirpairs[imunu][1]; 

      _cm_eq_zero ( Gr[ix][imunu] );

      /********************************************************************
       *             ------<------
       *             |           |
       *             |           ^
       *             |           |
       *             x----x+mu----
       *
       *                   +
       *
       *             x----x+mu----
       *             |           |
       *             |           ^
       *             |           |
       *             ------>-----
       *
       ********************************************************************/

      _cm_eq_cm_pl_cm ( U1, rectangles[g_iup[ix][imu]][imunu][0][0], rectangles[g_iup[ix][imu]][imunu][0][1] );
      /* _cm_eq_cm ( U1, rectangles[g_iup[ix][imu]][imunu][0][0] );
      _cm_eq_cm ( U1, rectangles[g_iup[ix][imu]][imunu][0][1]); */
      _cm_eq_cm_ti_cm ( U2, gauge_field + _GGI(ix,imu), U1 );
      _cm_eq_cm_ti_cm_dag ( U1, U2, gauge_field + _GGI(ix,imu) );
      _cm_pl_eq_cm ( Gr[ix][imunu], U1 );
#if 0
#endif  /* of if 0 */

      /********************************************************************
       * ------<------
       * |           |
       * |           ^
       * |           |
       * -----x-mu---x
       *
       *      +
       *
       * -----x-mu---x
       * |           |
       * |           ^
       * |           |
       * ------>-----
       *
       *
       ********************************************************************/

      _cm_eq_cm_pl_cm ( U1, rectangles[g_idn[ix][imu]][imunu][0][0], rectangles[g_idn[ix][imu]][imunu][0][1] );
      /* _cm_eq_cm ( U1, rectangles[g_idn[ix][imu]][imunu][0][0]  );
      _cm_eq_cm ( U1, rectangles[g_idn[ix][imu]][imunu][0][1]  ); */
      _cm_eq_cm_ti_cm ( U2, U1, gauge_field + _GGI( g_idn[ix][imu], imu ) );
      _cm_eq_cm_dag_ti_cm ( U1, gauge_field + _GGI( g_idn[ix][imu], imu ), U2 );
      _cm_pl_eq_cm ( Gr[ix][imunu], U1 );
#if 0
#endif  /* of if 0 */

      /********************************************************************
       * ---<---      ----<---
       * |     |      |      |
       * |     |      |      |
       * |     x+nu + x+nu   ^
       * |     |      |      |
       * |     |      |      |
       * --->--x      x--->---
       *
       *
       *
       ********************************************************************/

      _cm_eq_cm_pl_cm ( U1, rectangles[g_iup[ix][inu]][imunu][1][0], rectangles[g_iup[ix][inu]][imunu][1][1] );
      /* _cm_eq_cm ( U1, rectangles[g_iup[ix][inu]][imunu][1][0]  );
      _cm_eq_cm ( U1, rectangles[g_iup[ix][inu]][imunu][1][1]  ); */
      _cm_eq_cm_ti_cm ( U2, gauge_field + _GGI(ix,inu), U1 );
      _cm_eq_cm_ti_cm_dag ( U1, U2, gauge_field + _GGI(ix,inu) );
      _cm_pl_eq_cm ( Gr[ix][imunu], U1 );
#if 0
#endif  /* of if 0 */

      /********************************************************************
       *
       *
       *
       * ---<--x      x---<---
       * |     |      |      |
       * |     |      |      |
       * |     x-nu + x-nu   ^
       * |     |      |      |
       * |     |      |      |
       * --->---      ---->---
       *
       ********************************************************************/

      _cm_eq_cm_pl_cm ( U1, rectangles[g_idn[ix][inu]][imunu][1][0], rectangles[g_idn[ix][inu]][imunu][1][1] );
      /* _cm_eq_cm ( U1, rectangles[g_idn[ix][inu]][imunu][1][0] );
      _cm_eq_cm ( U1, rectangles[g_idn[ix][inu]][imunu][1][1] ); */
      _cm_eq_cm_ti_cm ( U2, U1, gauge_field + _GGI( g_idn[ix][inu], inu) );
      _cm_eq_cm_dag_ti_cm ( U1, gauge_field + _GGI( g_idn[ix][inu],inu), U2 );
      _cm_pl_eq_cm ( Gr[ix][imunu], U1 );
#if 0
#endif  /* of if 0 */

      /********************************************************************
       * anti-hermitean part and normalization
       ********************************************************************/
      _cm_eq_antiherm_cm ( U1, Gr[ix][imunu] );
      _cm_eq_cm_ti_re ( Gr[ix][imunu], U1, one_over_eight );

      /********************************************************************
       * at high verbosity write G_rect
       ********************************************************************/
      if ( g_verbose > 4 ) {
        for( int ia = 0; ia < 9; ia++ ) {
          fprintf ( stdout, "Gr %3d %3d %3d %3d    %d %d    %d %d    %25.16e %25.16e\n",
               ix                           / (LX*LY*LZ) + g_proc_coords[0]*T,
              (ix            % (LX*LY*LZ) ) / (LY*LZ)    + g_proc_coords[1]*LX,
              (ix            % (LY*LZ)    ) / (LZ)       + g_proc_coords[2]*LY,
              (ix            % LZ         )              + g_proc_coords[3]*LZ,
              imu, inu, ia/3, ia%3,
              Gr[ix][imunu][2*ia], Gr[ix][imunu][2*ia+1] );
        }
        fprintf ( stdout, "# Gr\n" );
      }


    }  /* end of loop on hyperplanes */
  }  /* end of loop on volume */

#if 0
  /********************************************************************
   * TEST
   * alternative calculation of rectangles
   ********************************************************************/
#ifndef HAVE_MPI
  FILE *fs = fopen( "gr.comp", "w" );

  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {

    double U1[18], U2[18];
    double R[6][18];

    int imunu = 0;

    for ( int imunu = 0; imunu < 6; imunu++ ) {
    
      const int imu = dirpairs[imunu][0];
      const int inu = dirpairs[imunu][1];

      memset ( R[imunu], 0, 18 * sizeof( double ) );

      /* 2mu + nu  */

      /* x -> x+mu -> x+mu+mu */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(g_iup[ix][imu],imu) );
      /* x -> x+mu -> x+mu+mu -> x+mu+mu+nu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_iup[ix][imu] ][imu], inu) );
      /* x -> x+mu -> x+mu+mu -> x+mu+mu+nu -> x+mu+nu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_iup[g_iup[ix][inu]][imu], imu) );
      /* x -> x+mu -> x+mu+mu -> x+mu+mu+nu -> x+mu+nu -> x+nu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], imu) );
      /* x -> x+mu -> x+mu+mu -> x+mu+mu+nu -> x+mu+nu -> x+nu -> x */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( ix, inu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* 2mu - nu  */

      /* x -> x-nu -> x-nu+mu */
      _cm_eq_cm_dag_ti_cm( U1, gauge_field+_GGI( g_idn[ix][inu], inu ), gauge_field+_GGI( g_idn[ ix][inu], imu) );
      /* x -> x-nu -> x-nu+mu -> x-nu+mu+mu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_iup[ix][imu] ][inu], imu) );
      /* x -> x-nu -> x-nu+mu -> x-nu+mu+mu -> x+mu+mu */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ g_iup[ g_iup[ix][imu] ][imu] ][inu], inu) );
      /* x -> x-nu -> x-nu+mu -> x-nu+mu+mu -> x+mu+mu -> x+mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][imu], imu) );
      /* x -> x-nu -> x-nu+mu -> x-nu+mu+mu -> x+mu+mu -> x+mu -> x */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( ix, imu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* -2mu + nu */

      /* x -> x+nu -> x+nu-mu */
      _cm_eq_cm_ti_cm_dag( U1, gauge_field+_GGI( ix, inu ), gauge_field+_GGI (g_idn[g_iup[ix][inu] ][imu], imu) );
      /* x -> x+nu -> x+nu-mu -> x+nu-mu-mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ g_iup[ix][inu] ][imu] ][imu], imu) );
      /* x -> x+nu -> x+nu-mu -> x+nu-mu-mu -> x-mu-mu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][imu], inu) );
      /* x -> x+nu -> x+nu-mu -> x+nu-mu-mu -> x-mu-mu -> x-mu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][imu], imu) );
      /* x -> x+nu -> x+nu-mu -> x+nu-mu-mu -> x-mu-mu -> x-mu -> x-mu */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ix][imu], imu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* -2mu - nu */

      /* ( x-mu -> x-mu-mu -> x-mu-mu-nu )^+ */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI( g_idn[ g_idn[g_idn[ix][imu] ][imu] ][inu],inu), gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][imu], imu) );
      /* ( x -> x-mu -> x-mu-mu -> x-mu-mu-nu )^+ */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ix][imu], imu) );
      /* x -> x-mu -> x-mu-mu -> x-mu-mu-nu -> x-mu-nu */
      _cm_eq_cm_dag_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ g_idn[g_idn[ix][inu] ][imu] ][imu], imu) );
      /* x -> x-mu -> x-mu-mu -> x-mu-mu-nu -> x-mu-nu -> x-nu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][inu], imu) );
      /* x -> x-mu -> x-mu-mu -> x-mu-mu-nu -> x-mu-nu -> x-nu -> x */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ix][inu], inu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* mu + 2nu */

      /* x -> x+mu -> x+mu+nu */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(g_iup[ix][imu],inu) );
      /* x -> x+mu -> x+mu+nu -> x+mu+nu+nu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_iup[ix][imu] ][inu], inu) );
      /* x -> x+mu -> x+mu+nu -> x+mu+nu+nu -> x+nu+nu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_iup[g_iup[ix][inu]][inu], imu) );
      /* x -> x+mu -> x+mu+nu -> x+mu+nu+nu -> x+nu+nu -> x+nu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], inu) );
      /* x -> x+mu -> x+mu+nu -> x+mu+nu+nu -> x+nu+nu -> x+nu -> x */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( ix, inu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* mu - 2nu */

      /* ( x -> x-nu -> x-nu-nu )^+ */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI( g_idn[ g_idn[ix][inu] ][inu], inu ), gauge_field+_GGI( g_idn[ ix][inu], inu) );
      /* x -> x-nu -> x-nu-nu -> x-nu-nu+mu */
      _cm_eq_cm_dag_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][inu] ][inu], imu) );
      /* x -> x-nu -> x-nu-nu -> x-nu-nu+mu -> x-nu+mu */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_iup[ g_idn[ g_idn[ix][inu] ][inu] ][imu], inu) );
      /* x -> x-nu -> x-nu-nu -> x-nu-nu+mu -> x-nu+mu -> x+mu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_idn[ix][inu] ][imu], inu) );
      /* x -> x-nu -> x-nu-nu -> x-nu-nu+mu -> x-nu+mu -> x+mu -> x */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( ix, imu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* -mu + 2nu */

      /* x -> x+nu -> x+nu+nu */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI( ix, inu ), gauge_field+_GGI ( g_iup[ix][inu], inu) );
      /* x -> x+nu -> x+nu+nu -> x+nu+nu-mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ g_iup[ g_iup[ix][inu] ][inu] ][imu], imu) );
      /* x -> x+nu -> x+nu+nu -> x+nu+nu-mu -> x+nu-mu */
      _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_iup[ g_idn[ix][imu] ][inu], inu) );
      /* x -> x+nu -> x+nu+nu -> x+nu+nu-mu -> x+nu-mu -> x-mu */
      _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ix][imu], inu) );
      /* x -> x+nu -> x+nu+nu -> x+nu+nu-mu -> x+nu-mu -> x-mu -> x */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ix][imu], imu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      /* -mu - 2nu */

      /* ( x-mu -> x-mu-nu -> x-mu-nu-nu )^+ */
      _cm_eq_cm_ti_cm( U1, gauge_field+_GGI( g_idn[ g_idn[g_idn[ix][imu] ][inu] ][inu],inu), gauge_field+_GGI( g_idn[ g_idn[ix][imu] ][inu], inu) );
      /* ( x -> x-mu -> x-mu-nu -> x-mu-nu-nu )^+ */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ix][imu], imu) );
      /* x -> x-mu -> x-mu-nu -> x-mu-nu-nu -> x-nu-nu */
      _cm_eq_cm_dag_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ g_idn[g_idn[ix][imu] ][inu] ][inu], imu) );
      /* x -> x-mu -> x-mu-nu -> x-mu-nu-nu -> x-nu-nu -> x-nu */
      _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_idn[ g_idn[ix][inu] ][inu], inu) );
      /* x -> x-mu -> x-mu-nu -> x-mu-nu-nu -> x-nu-nu -> x-nu -> x */
      _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_idn[ix][inu], inu) );
      _cm_pl_eq_cm ( R[imunu], U1 );

      _cm_eq_antiherm_cm ( U1, R[imunu] );
      _cm_eq_cm_ti_re ( R[imunu], U1, one_over_eight );

    }
    for ( int imunu = 0; imunu < 6; imunu++ ) {
      for ( int k = 0; k < 9; k++ ) {
          fprintf ( fs, "%6d %d    %d %d    %25.16e %25.16e     %25.16e%25.16e\n", ix, imunu, k/3, k%3,
              Gr[ix][imunu][2*k], Gr[ix][imunu][2*k+1],
              R[imunu][2*k], R[imunu][2*k+1] );
      }
    }
  }  /* end of loop on ix */
  fclose ( fs );
#endif  /* of ifndef HAVE_MPI */

  /********************************************************************
   * end of TEST
   ********************************************************************/
#endif  /* of if 0 */

  fini_5level_dtable ( &rectangles );

  fini_3level_dtable ( &plaquettes );

#ifdef HAVE_MPI
  mpi_fini_xchanger ( &x_plaq );
  mpi_fini_xchanger ( &x_rect );
#endif
  return( 0 );

}  /* end of G_plaq_rect */


/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 ****************************************************************************/
int gluonic_operators ( double ** op, double * const gfield ) {

  unsigned int const VOL3 = LX * LY * LZ;
  double ** pl = init_2level_dtable ( T, 2 );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double s[18], t[18], u[18];
    double pl_tmp[2] = { 0, 0. };

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int iy = 0; iy < VOL3; iy++) {

      unsigned int const ix = it * VOL3 + iy;

      /* time-like */
      for ( int nu = 1; nu < 4; nu++ ) {
        _cm_eq_cm_ti_cm(s, gfield + _GGI(ix, 0), gfield + _GGI(g_iup[ix][ 0], nu) );
        _cm_eq_cm_ti_cm(t, gfield + _GGI(ix,nu), gfield + _GGI(g_iup[ix][nu],  0) );
        _cm_eq_cm_ti_cm_dag(u, s, t);
        _re_pl_eq_tr_cm ( &(pl_tmp[0]), u );
      }

      /* space-like */
      for ( int mu = 1; mu<3; mu++) {
      for ( int nu = mu+1; nu < 4; nu++) {
        _cm_eq_cm_ti_cm(s, gfield + _GGI(ix, mu), gfield + _GGI( g_iup[ix][mu], nu) );
        _cm_eq_cm_ti_cm(t, gfield + _GGI(ix, nu), gfield + _GGI( g_iup[ix][nu], mu) );
        _cm_eq_cm_ti_cm_dag(u, s, t);
        _re_pl_eq_tr_cm( &(pl_tmp[1]), u);
      }}
    }

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    pl[it][0] += pl_tmp[0];
    pl[it][1] += pl_tmp[1];

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double ** buffer = init_2level_dtable ( T, 2 );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl[0], buffer[0], 2*T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer[0], 2*T, MPI_DOUBLE, op[0], 2*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_2level_dtable ( &buffer );
#else
  memcpy ( op[0], pl[0], 2*T_global*sizeof(double) );
#endif

  fini_2level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators */

/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 * calculated from gluon field strength tensor G
 *
 * 6 components of G expected; each 3x3 complex matrix
 * 
 * G_{0,1}  G_{0,2}   G_{0,3}   G_{1,2}   G_{1,3}   G_{2,3}
 *   0        1         2         3         4         5
 *
 * further we asume anti-symmetry in mu, nu
 * G_{1,0} = - G_{0,1}
 * G_{2,0} = - G_{0,2}
 * G_{3,0} = - G_{0,3}
 * G_{2,1} = - G_{1,2}
 * G_{3,1} = - G_{1,3}
 * G_{3,2} = - G_{2,3}
 *
 * diagonal elements are zero
 ****************************************************************************/
int gluonic_operators_eo_from_fst ( double ** op, double *** const G ) {

  unsigned int const VOL3 = LX * LY * LZ;
  double ** pl = init_2level_dtable ( T, 2 );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double s[18];
    double pl_tmp[2] = { 0, 0. };

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int iy = 0; iy < VOL3; iy++) {

      unsigned int const ix = it * VOL3 + iy;

      /* for O44 : G_{0,1} x G_{1,0} + G_{0,2} x G_{2,0} + G_{0,3} x G_{3,0} 
       * indices        0  x      0         1  x      1         2  x      2  */

      for ( int nu = 0; nu < 3; nu++ ) {
        _cm_eq_cm_ti_cm ( s, G[ix][nu], G[ix][nu] );
        _re_pl_eq_tr_cm ( &(pl_tmp[0]), s );
      }

      /* for Okk : G_{1,2} x G_{1,2} + G_{1,2} x G_{1,2} + G_{1,2} x G_{1,2} 
       * indices        3  x      3         4  x      4         5  x      5 */
      for ( int nu = 3; nu<6; nu++) {
        _cm_eq_cm_ti_cm ( s, G[ix][nu], G[ix][nu] );
        _re_pl_eq_tr_cm ( &(pl_tmp[1]), s );
      }
    }
    pl_tmp[0] *= -1.;
    pl_tmp[1] *= -1.;

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    pl[it][0] += pl_tmp[0];
    pl[it][1] += pl_tmp[1];

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double ** buffer = init_2level_dtable ( T, 2 );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl[0], buffer[0], 2*T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer[0], 2*T, MPI_DOUBLE, op[0], 2*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_2level_dtable ( &buffer );
#else
  memcpy ( op[0], pl[0], 2*T_global*sizeof(double) );
#endif

  fini_2level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators_from_fst */


/********************************************************************
 * out
 *   Gp : non-zero field strength tensor components
 *        from plaquettes, 1x1 loops
 * in
 *   antihermitean : hermitean = 0, antihermitean = 1
 ********************************************************************/
int G_plaq ( double *** Gp, double * const gauge_field, int const antihermitean ) {

  const int dirpairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
  const double one_over_four  = 0.250;

#ifdef HAVE_MPI
  xchanger_type x_plaq;
  mpi_init_xchanger ( &x_plaq, 18 );
#endif

  double ** plaquettes = init_2level_dtable ( VOLUMEPLUSRAND, 18 );
  if ( plaquettes == NULL ) {
    fprintf ( stderr, "[G_plaq] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

#ifdef HAVE_MPI
  /********************************************************************
   * exchange gauge field to be sure
   ********************************************************************/
  xchange_gauge_field ( gauge_field );
#endif

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
  omp_init_lock(&writelock);
#endif
  double symact = 0.;
#endif

  memset ( Gp[0][0], 0, VOLUME * 54 * sizeof ( double ) );

  /********************************************************************
   * calculate elementary plaquettes for all triples (ix, mu, nu )
   ********************************************************************/

  for ( int imunu = 0; imunu < 6; imunu++) {

    const int imu = dirpairs[imunu][0]; 
    const int inu = dirpairs[imunu][1]; 

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
  
      double U1[18], U2[18];

        /********************************************************************
         * P <- [ U_mu(x) U_nu(x+mu) ] * [ U_nu(x) U_mu(x+nu) ]^+
         *   =  U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
         *
         * x+nu >    x+mu+nu
         *   _______
         *  |       |
         *  |       |
         * ^|       | 
         * _|_______|
         *  |x  <    x + mu
         *
         ********************************************************************/
        /********************************************************************
         * 2 corners,
         * (U1) U_mu(x) * U_nu(x+mu)
         * (U2) U_nu(x) * U_mu(x+nu)
         * then multply 
         * U1 x U2^+ = U_mu(x) * U_nu(x+mu) * U_mu(x+nu)^+ * U_nu(x)^+
         *           = U(x,x+mu) * U(x+mu,x+mu+nu) * U(x+nu+mu,x+nu) * U(x+nu,x)
         ********************************************************************/
        _cm_eq_cm_ti_cm ( U1, gauge_field + _GGI(ix, imu), gauge_field + _GGI( g_iup[ix][imu], inu) );
        _cm_eq_cm_ti_cm ( U2, gauge_field + _GGI(ix, inu), gauge_field + _GGI( g_iup[ix][inu], imu) );
        _cm_eq_cm_ti_cm_dag ( plaquettes[ix], U1, U2 );
 
    }  /* end of loop on VOLUME */

    /********************************************************************
     * xchange the plaquettes,
     *   INCLUDING edges
     ********************************************************************/
#ifdef HAVE_MPI
    mpi_xchanger ( plaquettes[0], &x_plaq );
#endif

  /********************************************************************
   * build G_munu from plaquettes
   ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(plaquettes)
#endif
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      double U1[18], U2[18], U3[18], U4[18];
    
      _cm_eq_zero ( U1 );
      
      /********************************************************************
       * (1) U1 += U_{mu,nu}(x)
       ********************************************************************/
      _cm_pl_eq_cm ( U1, plaquettes[ix] );

      /********************************************************************
       * (2) U1 +=  U_{mu,nu}(x - mu)
       ********************************************************************/
      _cm_eq_cm_ti_cm( U2, plaquettes[ g_idn[ix][imu] ], gauge_field + _GGI( g_idn[ix][imu], imu ) );
      _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][imu], imu ), U2 );
      _cm_pl_eq_cm ( U1, U3 );

      /********************************************************************
       * (3) U1 += U_{mu,nu}(x - nu)
       ********************************************************************/
      _cm_eq_cm_ti_cm( U2, plaquettes[ g_idn[ix][inu] ], gauge_field + _GGI( g_idn[ix][inu], inu ) );
      _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][inu], inu ), U2 );
      _cm_pl_eq_cm ( U1, U3 );

      /********************************************************************
       * (4) U1 += U_{mu,nu}(x - nu - mu)
       ********************************************************************/
      _cm_eq_cm_ti_cm ( U4, gauge_field + _GGI(g_idn[g_idn[ix][imu]][inu], imu), gauge_field + _GGI(g_idn[ix][inu], inu) );
      _cm_eq_cm_ti_cm ( U2, plaquettes[ g_idn[ g_idn[ix][inu] ][imu] ],  U4 );
      _cm_eq_cm_dag_ti_cm ( U3, U4, U2 );
      _cm_pl_eq_cm ( U1, U3 );

      _cm_ti_eq_re ( U1, one_over_four );

      if ( antihermitean == 0 ) {
        project_to_generators_hermitean ( Gp[ix][imunu], U1 );
      } else if ( antihermitean == 1 ) {
        project_to_generators ( Gp[ix][imunu], U1 );
      }

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
      omp_set_lock(&writelock);
#endif
      _re_pl_eq_tr_cm( &symact, U1 );
#ifdef HAVE_OPENMP
      omp_unset_lock(&writelock);
#endif
#endif

    }  /* end of loop on volume */

  }  /* end of loop on nu, mu */

  fini_2level_dtable ( &plaquettes );

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
  omp_destroy_lock(&writelock);
#endif
  symact /= (double)VOLUME * g_nproc * 18.;
#ifdef HAVE_MPI
  double dtmpexc = symact;
  if ( MPI_Allreduce ( &dtmpexc,  &symact, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[G_plaq] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }
#endif
  if ( g_cart_id == 0 ) {
    fprintf ( stdout, "# [G_plaq] plaquette action = %25.16e %s %d\n", symact, __FILE__, __LINE__ );
  }
#endif

#ifdef HAVE_MPI
  mpi_fini_xchanger ( &x_plaq );
#endif
  return( 0 );

}  /* end of G_plaq */


/********************************************************************
 * out
 *   Gr : non-zero field strength tensor components
 *        from rectangles, 1x2 and 2x1 loops  
 * in
 *   antihermitean : hermitean = 0, anti-hermitean = 1
 ********************************************************************/
int G_rect ( double *** Gr, double * const gauge_field, int const antihermitean ) {

  const int dirpairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
  const double one_over_eight = 0.125;

#ifdef HAVE_MPI
  xchanger_type x_rect;
  mpi_init_xchanger ( &x_rect, 18 );
#endif

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
  omp_init_lock(&writelock);
#endif
  double symact = 0.;
#endif

#ifdef HAVE_MPI
  /********************************************************************
   * exchange gauge field to be sure
   ********************************************************************/
  xchange_gauge_field ( gauge_field );
#endif

  /********************************************************************
   * now build the rectangles from products of plaquettes
   *
   * needs rotation of plaquettes taken from neighbours to current site
   ********************************************************************/
  double ** rectangles = init_2level_dtable ( VOLUMEPLUSRAND, 18 );
  if ( rectangles == NULL ) {
    fprintf ( stderr, "[G_rect] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }
  
  memset ( Gr[0][0], 0, VOLUME * 54 * sizeof ( double ) );

  for ( int imunu = 0; imunu < 6; imunu++) {

    const int imu = dirpairs[imunu][0]; 
    const int inu = dirpairs[imunu][1]; 

    for ( int idir = 0; idir < 2; idir++ ) {

      /********************************************************************
       * calculation of rectangles
       ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for ( unsigned int ix = 0; ix < VOLUME; ix++ )
      {

        double U1[18], U2[18];

        memset ( rectangles[ix], 0, 18 * sizeof( double ) );

        if ( idir == 0 ) {

          /* x -> x+mu -> x+mu+nu */
          _cm_eq_cm_ti_cm( U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(g_iup[ix][imu],inu) );
          /* x -> x+mu -> x+mu+nu -> x+nu */
          _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], imu) );
          /* x -> x+mu+nu -> x+nu -> x+nu-mu */
          _cm_eq_cm_ti_cm_dag( U1, U2, gauge_field+_GGI( g_idn[g_iup[ix][inu]][imu], imu) );
          /* x -> x+mu+nu -> x+nu -> x+nu-mu -> x-mu */
          _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_idn[ix][imu], inu) );
          /* x -> x+mu+nu -> x+nu -> x+nu-mu -> x-mu -> x */
          _cm_eq_cm_ti_cm( rectangles[ix], U2, gauge_field+_GGI( g_idn[ix][imu], imu) );

        } else if ( idir == 1 ) {

          /* x -> x-nu -> x-nu+mu */
          _cm_eq_cm_dag_ti_cm( U1, gauge_field+_GGI( g_idn[ix][inu],inu ), gauge_field+_GGI(g_idn[ix][inu],imu) );
          /* x -> x-nu -> x-nu+mu -> x+mu */
          _cm_eq_cm_ti_cm( U2, U1, gauge_field+_GGI( g_iup[ g_idn[ix][inu] ][imu], inu) );
          /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu */
          _cm_eq_cm_ti_cm( U1, U2, gauge_field+_GGI( g_iup[ix][imu], inu) );
          /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu -> x+nu */
         _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field+_GGI( g_iup[ix][inu], imu) );
          /* x -> x-nu -> x-nu+mu -> x+mu -> x+mu+nu -> x+nu -> x */
         _cm_eq_cm_ti_cm_dag( rectangles[ix], U2, gauge_field+_GGI( ix, inu) );

       }
      } /* end of loop on ix */

      /********************************************************************
       * xchange the rectangles
       ********************************************************************/
#ifdef HAVE_MPI
      mpi_xchanger ( rectangles[0], &x_rect );
#endif

      /********************************************************************
       * build G_munu from rectangles
       ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(rectangles)
#endif
      for ( unsigned int ix = 0; ix < VOLUME; ix++ )
      {
  
        double U1[18], U2[18], U3[18], RR[18];

        memset ( RR, 0, 18 * sizeof( double ) );

        if ( idir == 0 ) {

          /********************************************************************
           *             ------<------
           *             |           |
           *             |           ^
           *             |           |
           *             x----x+mu----
           *
           *                   +
           *
           *             x----x+mu----
           *             |           |
           *             |           ^
           *             |           |
           *             ------>-----
           *
           ********************************************************************/

          _cm_eq_cm_ti_cm( U1, gauge_field + _GGI(ix, imu), rectangles[g_iup[ix][imu]] );
          _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field + _GGI(ix, imu) );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */

          _cm_eq_cm_dag_ti_cm ( U3, gauge_field + _GGI( g_idn[ix][inu], inu ), gauge_field + _GGI( g_idn[ix][inu], imu ) );
          _cm_eq_cm_ti_cm ( U1, U3, rectangles[ g_idn[ g_iup[ix][imu]][inu] ] );
          _cm_eq_cm_ti_cm_dag ( U2, U1, U3 );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */
    
          /********************************************************************
           * ------<------
           * |           |
           * |           ^
           * |           |
           * -----x-mu---x
           *
           *      +
           *
           * -----x-mu---x
           * |           |
           * |           ^
           * |           |
           * ------>-----
           *
           *
           ********************************************************************/

          _cm_eq_cm_dag_ti_cm ( U1, gauge_field + _GGI( g_idn[ix][imu], imu), rectangles[g_idn[ix][imu]] );
          _cm_eq_cm_ti_cm ( U2, U1, gauge_field + _GGI( g_idn[ix][imu], imu) );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */
    
          _cm_eq_cm_ti_cm ( U3, gauge_field + _GGI( g_idn[ g_idn[ix][inu] ][imu], imu ), gauge_field + _GGI( g_idn[ix][inu], inu ) );
          _cm_eq_cm_dag_ti_cm ( U1, U3, rectangles[ g_idn[ g_idn[ix][imu] ][inu]  ] );
          _cm_eq_cm_ti_cm ( U2, U1, U3 );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */

        } else if ( idir == 1 ) {

          /********************************************************************
           *         ---<---      ----<---
           *         |     |      |      |
           *         |     |      |      |
           *   x-mu+nu     |    x+nu     ^
           *         |     |      |      |
           *         |     |      |      |
           *         --->--x      x--->---
           *
           *
           *
           ********************************************************************/

          _cm_eq_cm_ti_cm( U1, gauge_field + _GGI(ix,inu), rectangles[g_iup[ix][inu]] );
          _cm_eq_cm_ti_cm_dag( U2, U1, gauge_field + _GGI(ix,inu) );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */

          _cm_eq_cm_dag_ti_cm( U3, gauge_field + _GGI( g_idn[ix][imu], inu), gauge_field + _GGI( g_idn[ix][imu] , imu ) );
          _cm_eq_cm_ti_cm ( U1, rectangles[ g_iup[ g_idn[ix][imu] ][inu] ], U3 );
          _cm_eq_cm_dag_ti_cm ( U2, U3, U1 );
          _cm_pl_eq_cm ( RR, U2 );
#if 0   
#endif  /* of if 0 */
    
          /********************************************************************
           *
           *
           *
           *       ---<--x      x---<---
           *       |     |      |      |
           *       |     |      |      |
           * x-nu-mu     ^   x-nu      ^
           *       |     |      |      |
           *       |     |      |      |
           *       --->---      ---->---
           *
           ********************************************************************/

          _cm_eq_cm_dag_ti_cm ( U1, gauge_field + _GGI( g_idn[ix][inu], inu ), rectangles[g_idn[ix][inu] ] );
          _cm_eq_cm_ti_cm ( U2, U1, gauge_field + _GGI( g_idn[ix][inu], inu ) );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */


          _cm_eq_cm_ti_cm ( U3, gauge_field + _GGI ( g_idn[ g_idn[ix][inu] ][imu], inu ), gauge_field + _GGI( g_idn[ix][imu], imu ) );
          _cm_eq_cm_dag_ti_cm ( U1, U3, rectangles[ g_idn[ g_idn[ix][inu] ][imu] ] );
          _cm_eq_cm_ti_cm ( U2, U1, U3 );
          _cm_pl_eq_cm ( RR, U2 );
#if 0
#endif  /* of if 0 */

        }  /* end of if on idir */

        /********************************************************************
         * anti-hermitean part and normalization
         ********************************************************************/

        _cm_ti_eq_re ( RR, one_over_eight );

        double p[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
        if ( antihermitean == 0 ) {
          project_to_generators_hermitean ( p, RR );
        } else if ( antihermitean == 1 ) {
          project_to_generators ( p, RR );
        }
 
        Gr[ix][imunu][0] += p[0];
        Gr[ix][imunu][1] += p[1];
        Gr[ix][imunu][2] += p[2];
        Gr[ix][imunu][3] += p[3];
        Gr[ix][imunu][4] += p[4];
        Gr[ix][imunu][5] += p[5];
        Gr[ix][imunu][6] += p[6];
        Gr[ix][imunu][7] += p[7];
        Gr[ix][imunu][8] += p[8];

        if ( g_verbose > 4 ) {
          double U[18], q[9];
          memcpy ( q, p, 9*sizeof(double) );
          if ( antihermitean == 0 ) {
            q[0] = 0.;
            restore_from_generators_hermitean ( U, q );
          } else if ( antihermitean == 1 ) {
            restore_from_generators ( U, q );
          }
          const int dir[3] = { imu, (1-idir)*imu + idir*inu, inu };

          for( int ia = 0; ia < 9; ia++ ) {
            fprintf ( stdout, "RR-tl %3d %3d %3d %3d    %d %d %d     %d %d     %25.16e %25.16e\n",
                 ix                           / (LX*LY*LZ) + g_proc_coords[0]*T,
                (ix            % (LX*LY*LZ) ) / (LY*LZ)    + g_proc_coords[1]*LX,
                (ix            % (LY*LZ)    ) / (LZ)       + g_proc_coords[2]*LY,
                (ix            % LZ         )              + g_proc_coords[3]*LZ,
                dir[0], dir[1], dir[2], ia/3, ia%3, U[2*ia], U[2*ia+1] );
          }
          fprintf ( stdout, "# RR-tl\n" );
        }

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
        omp_set_lock(&writelock);
#endif
        _re_pl_eq_tr_cm( &symact, RR );
#ifdef HAVE_OPENMP
        omp_unset_lock(&writelock);
#endif
#endif

      }  /* end of loop on volume */
  
    }  /* end of loop on idir */

  }  /* end of loop on mu nu  */

  fini_2level_dtable ( &rectangles );

#ifdef _SYM_ACTION
#ifdef HAVE_OPENMP
  omp_destroy_lock(&writelock);
#endif
  symact /= (double)VOLUME * g_nproc * 18;
#ifdef HAVE_MPI
  double dtmpexc = symact;
  if ( MPI_Allreduce ( &dtmpexc,  &symact, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[G_rect] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }
#endif
  if ( g_cart_id == 0 ) {
    fprintf ( stdout, "# [G_rect] rectangle action = %25.16e %s %d\n", symact, __FILE__, __LINE__ );
  }
#endif


#ifdef HAVE_MPI
  mpi_fini_xchanger ( &x_rect );
#endif
  return( 0 );

}  /* end of G_rect */

/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 * calculated from gluon field strength tensor G
 *
 * 6 components of G expected; each 3x3 complex matrix
 * 
 * G_{0,1}  G_{0,2}   G_{0,3}   G_{1,2}   G_{1,3}   G_{2,3}
 *   0        1         2         3         4         5
 *
 * further we asume anti-symmetry in mu, nu
 * G_{1,0} = - G_{0,1}
 * G_{2,0} = - G_{0,2}
 * G_{3,0} = - G_{0,3}
 * G_{2,1} = - G_{1,2}
 * G_{3,1} = - G_{1,3}
 * G_{3,2} = - G_{2,3}
 *
 * diagonal elements are zero
 ****************************************************************************/
int gluonic_operators_eo_from_fst_projected ( double ** op, double *** const G, int const traceless ) {

  unsigned int const VOL3 = LX * LY * LZ;
  double ** pl = init_2level_dtable ( T, 2 );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators_eo_from_fst_projected] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double pl_tmp[2] = { 0, 0. };

    if ( traceless ) {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;

        /* for O44 : G_{0,1} x G_{1,0} + G_{0,2} x G_{2,0} + G_{0,3} x G_{3,0} 
         * indices        0  x      0         1  x      1         2  x      2  */

        for ( int nu = 0; nu < 3; nu++ ) {
          for ( int k = 1; k < 9; k++ ) {
            pl_tmp[0] += G[ix][nu][k] * G[ix][nu][k];
          }
        }

        /* for Okk : G_{1,2} x G_{1,2} + G_{1,3} x G_{1,3} + G_{2,3} x G_{2,3} 
         * indices        3  x      3         4  x      4         5  x      5 */
        for ( int nu = 3; nu<6; nu++) {
          for ( int k = 1; k < 9; k++ ) {
            pl_tmp[1] += G[ix][nu][k] * G[ix][nu][k];
          }
        }
      }
    } else {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;

        /* for O44 : G_{0,1} x G_{1,0} + G_{0,2} x G_{2,0} + G_{0,3} x G_{3,0} 
         * indices        0  x      0         1  x      1         2  x      2  */

        for ( int nu = 0; nu < 3; nu++ ) {
          for ( int k = 0; k < 9; k++ ) {
            pl_tmp[0] += G[ix][nu][k] * G[ix][nu][k];
          }
        }

        /* for Okk : G_{1,2} x G_{1,2} + G_{1,3} x G_{1,3} + G_{2,3} x G_{2,3} 
         * indices        3  x      3         4  x      4         5  x      5 */
        for ( int nu = 3; nu<6; nu++) {
          for ( int k = 0; k < 9; k++ ) {
            pl_tmp[1] += G[ix][nu][k] * G[ix][nu][k];
          }
        }
      }
    }

    pl_tmp[0] *= 0.5;
    pl_tmp[1] *= 0.5;

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    pl[it][0] += pl_tmp[0];
    pl[it][1] += pl_tmp[1];

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double ** buffer = init_2level_dtable ( T, 2 );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators_eo_from_fst_projected] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl[0], buffer[0], 2*T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_eo_from_fst_projected] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer[0], 2*T, MPI_DOUBLE, op[0], 2*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_eo_from_fst_projected] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_2level_dtable ( &buffer );
#else
  memcpy ( op[0], pl[0], 2*T_global*sizeof(double) );
#endif

#ifdef _SYM_ACTION
  if ( g_cart_id == 0 ) {
    double dtmp[2] = { 0., 0. };
    for ( int i =0; i<T_global; i++ ) {
      dtmp[0] += op[i][0];
      dtmp[1] += op[i][1];
    }
    dtmp[0] /= (double)VOLUME * 18.;
    dtmp[1] /= (double)VOLUME * 18.;
    fprintf ( stdout, "# [gluonic_operators_eo_from_fst_projected] FST action temporal %25.16e   spatial %25.16e   total %25.16e\n", dtmp[0], dtmp[1], dtmp[0]+dtmp[1] );
  }
#endif

  fini_2level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators_from_fst_projected */

/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 * use projected fields
 ****************************************************************************/
int gluonic_operators_projected ( double ** const op, double *** const G ) {

  double const sqrt_three_over_two = 1.22474487139158904909;

  unsigned int const VOL3 = LX * LY * LZ;
  double ** pl = init_2level_dtable ( T, 2 );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double pl_tmp[2] = { 0, 0. };

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int iy = 0; iy < VOL3; iy++) {

      unsigned int const ix = it * VOL3 + iy;

      /* time-like */
      for ( int nu = 0; nu < 3; nu++ ) {
        pl_tmp[0] += G[ix][nu][0];
      }

      /* space-like */
      for ( int nu = 3; nu < 6; nu++) {
        pl_tmp[1] += G[ix][nu][0];
      }
    }

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    pl[it][0] += pl_tmp[0] * sqrt_three_over_two;
    pl[it][1] += pl_tmp[1] * sqrt_three_over_two;

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double ** buffer = init_2level_dtable ( T, 2 );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl[0], buffer[0], 2*T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer[0], 2*T, MPI_DOUBLE, op[0], 2*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_2level_dtable ( &buffer );
#else
  memcpy ( op[0], pl[0], 2*T_global*sizeof(double) );
#endif

  fini_2level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators_projected */


/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 * calculated from gluon field strength tensor G
 *
 * 6 components of G expected; each 3x3 complex matrix
 * 
 * G_{0,1}  G_{0,2}   G_{0,3}   G_{1,2}   G_{1,3}   G_{2,3}
 *   0        1         2         3         4         5
 *
 * further we asume anti-symmetry in mu, nu
 * G_{1,0} = - G_{0,1}
 * G_{2,0} = - G_{0,2}
 * G_{3,0} = - G_{0,3}
 * G_{2,1} = - G_{1,2}
 * G_{3,1} = - G_{1,3}
 * G_{3,2} = - G_{2,3}
 *
 * diagonal elements are zero
 ****************************************************************************/
int gluonic_operators_qtop_from_fst_projected ( double * op, double *** const G, int const traceless ) {

  double const qtop_norm = -1. / ( 8. * M_PI * M_PI );
  unsigned int const VOL3 = LX * LY * LZ;
  double * pl = init_1level_dtable ( T );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators_eo_from_fst_projected] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double pl_tmp = 0.;

    if ( traceless ) {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;

        /* for qtop   : G_{0,1} x G_{2,3} - G_{0,2} x G_{1,3} + G_{0,3} x G_{1,2} 
         * indices      0       x 5       - 1       x 4       + 2       x 3        */

        for ( int k = 1; k < 9; k++ ) {
          pl_tmp +=   G[ix][0][k] * G[ix][5][k]
                    - G[ix][1][k] * G[ix][4][k]
                    + G[ix][2][k] * G[ix][3][k];
        }
      }
    } else {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;

        /* for O44 : G_{0,1} x G_{1,0} + G_{0,2} x G_{2,0} + G_{0,3} x G_{3,0} 
         * indices        0  x      0         1  x      1         2  x      2  */

        for ( int k = 0; k < 9; k++ ) {
          pl_tmp +=   G[ix][0][k] * G[ix][5][k]
                    - G[ix][1][k] * G[ix][4][k]
                    + G[ix][2][k] * G[ix][3][k];
        }
      }
    }

    pl_tmp *= qtop_norm;

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    pl[it] += pl_tmp;

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double * buffer = init_1level_dtable ( T );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators_qtop_from_fst_projected] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl, buffer, T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_qtop_from_fst_projected] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer, T, MPI_DOUBLE, op, T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_qtop_from_fst_projected] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_1level_dtable ( &buffer );
#else
  memcpy ( op, pl, T_global*sizeof(double) );
#endif

  fini_1level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators_qtop_from_fst_projected */



/****************************************************************************/
/****************************************************************************/

/****************************************************************************
 * operators for gluon momentum fraction
 * calculated from gluon field strength tensor G
 *
 * 6 components of G expected; each 3x3 complex matrix
 * 
 * G_{0,1}  G_{0,2}   G_{0,3}   G_{1,2}   G_{1,3}   G_{2,3}
 *   0        1         2         3         4         5
 *
 * further we asume anti-symmetry in mu, nu
 * G_{1,0} = - G_{0,1}
 * G_{2,0} = - G_{0,2}
 * G_{3,0} = - G_{0,3}
 * G_{2,1} = - G_{1,2}
 * G_{3,1} = - G_{1,3}
 * G_{3,2} = - G_{2,3}
 *
 * diagonal elements are zero
 *
 * all non-zero tensor components of
 * G_mu,nu G_alpha,beta
 * taking into account symmetry 
 * in total 21
 *
 * the stored sequence is
 *
 * (0,1) x (0,1)   0
 * (0,1) x (0,2)   1
 * (0,1) x (0,3)   2
 * (0,1) x (1,2)   3
 * (0,1) x (1,3)   4
 * (0,1) x (2,3)   5
 *
 * (0,2) x (0,2)   6
 * (0,2) x (0,3)   7
 * (0,2) x (1,2)   8
 * (0,2) x (1,3)   9
 * (0,2) x (2,3)  10
 *
 * (0,3) x (0,3)  11
 * (0,3) x (1,2)  12
 * (0,3) x (1,3)  13
 * (0,3) x (2,3)  14
 *
 * (1,2) x (1,2)  15
 * (1,2) x (1,3)  16
 * (1,2) x (2,3)  17
 *
 * (1,3) x (1,3)  18
 * (1,3) x (2,3)  19
 *
 * (2,3) x (2,3)  20
 *
 * Construction of Gluon EMt elements
 *
 * T_00 = G_0,mu G_0,mu
 *      = G_01 G_01 + G_02 G_02 + G_03 G_03 =  0 +  6 + 11 
 *
 * T_01 = G_0,mu G_1,mu
 *      = G_02 G_12 + G_03 G_13 = + 8 + 13
 *
 * T_02 = G_0,mu G_2,mu
 *      = G_01 G_21 + G_03 G_23 = - 3 + 14
 *
 * T_03 = G_0,mu G_3,mu
 *      = G_01 G_31 + G_02 G_32 = - 4 - 10
 *
 * T_11 = G_1,mu G_1,mu
 *      = G_10 G_10 + G_12 G_12 + G_13 G_13 =  0 + 15 + 18
 *
 * T_12 = G_1,mu G_2,mu
 *      = G_10 G_20 + G_13 G_23 = + 1 + 19
 *
 * T_13 = G_1,mu G_3,mu
 *      = G_10 G_30 + G_12 G_32 = + 2 - 17
 *
 * T_22 = G_2,mu G_2,mu
 *      = G_20 G_20 + G_21 G_21 + G_23 G_23 =  6 + 15 + 20
 *
 * T_23 = G_2,mu G_3,mu
 *      = G_20 G_30 + G_21 G_31 = + 7 + 16
 *
 * T_33 = G_3,mu G_3,mu
 *      = G_30 G_30 + G_31 G_31 + G_32 G_32 = 11 + 18 + 20
 *
 *
 ****************************************************************************/
int gluonic_operators_gg_from_fst_projected ( double ** op, double *** const G, int const traceless ) {

  unsigned int const VOL3 = LX * LY * LZ;
  double ** pl = init_2level_dtable ( T, 21 );
  if ( pl == NULL ) {
    fprintf( stderr, "[gluonic_operators_eo_from_fst_projected] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  for ( int it = 0; it < T; it++ ) {
#ifdef HAVE_OPENMP
    omp_init_lock(&writelock);

#pragma omp parallel shared(it)
{
#endif
    double pl_tmp[21];
    memset( pl_tmp, 0, 21*sizeof(double) );

    if ( traceless ) {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;


        int icount = 0;
        for ( int ia =  0; ia < 6; ia++ ) {
        for ( int ib = ia; ib < 6; ib++ ) {

          for ( int k = 1; k < 9; k++ ) {
            pl_tmp[ icount ] += G[ix][ia][k] * G[ix][ib][k];
          }
          icount++;

        }}

      }  /* end of loop on VOL3 */

    } else {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int iy = 0; iy < VOL3; iy++) {

        unsigned int const ix = it * VOL3 + iy;

        int icount = 0;
        for ( int ia = 0; ia < 6; ia++ ) {
        for ( int ib = ia; ib < 6; ib++ ) {

          for ( int k = 0; k < 9; k++ ) {
            pl_tmp[ icount ] += G[ix][ia][k] * G[ix][ib][k];
          }
          icount++;
        }}

      }  /* end of loop on VOL3  */
    
    }  /* end of if on traceless */

#ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
#endif

    for ( int i = 0; i < 21; i++ ) pl[it][i] += pl_tmp[i];

#ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
}  /* end of parallel region */
    omp_destroy_lock(&writelock);
#endif

  }  /* end of loop on timeslices */


#ifdef HAVE_MPI

  double ** buffer = init_2level_dtable ( T, 21 );
  if ( buffer == NULL ) {
    fprintf( stderr, "[gluonic_operators_eo_from_fst_projected] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(3);
  }
  if ( MPI_Reduce ( pl[0], buffer[0], 21*T, MPI_DOUBLE, MPI_SUM,  0, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_eo_from_fst_projected] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( MPI_Gather ( buffer[0], 21*T, MPI_DOUBLE, op[0], 21*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[gluonic_operators_eo_from_fst_projected] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  fini_2level_dtable ( &buffer );
#else
  memcpy ( op[0], pl[0], 21*T_global*sizeof(double) );
#endif

  fini_2level_dtable ( &pl );
  return( 0 );

}  /* end of gluonic_operators_gg_from_fst_projected */

}  /* end of namespace cvc */
