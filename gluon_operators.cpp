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

/********************************************************************/
/********************************************************************/

namespace cvc {

/********************************************************************
 * out
 *   Gp : non-zero field strength tensor components
 *        from plaquettes, 1x1 loops
 *   Gr : non-zero field strength tensor components
 *        from rectangles, 1x2 and 2x1 loops  
 ********************************************************************/
int G_plaq_rect ( double *** Gp, double *** Gr, double * const gauge_field) {

  const int dirpairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
  const int dirpairsinv[4][4] = { 
      { -1,  0,  1,  2},
      {  0, -1,  3,  4},
      {  1,  3, -1,  5},
      {  2,  4,  5, -1} };
 
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
    }
  }

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
   ********************************************************************/
  double ***** rectangles = init_5level_dtable ( VOLUMEPLUSRAND, 6, 2, 2, 18 );
  if ( rectangles == NULL ) {
    fprintf ( stderr, "[clover_rectangle_term] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

#pragma omp parallel for shared(plaquettes)
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
#if 0   
#endif   
    }  /* end of loop on imunu  */

  }  /* end of loop on ix */

#if 0
  /********************************************************************
   * TEST
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

  STOPPED HERE
  /********************************************************************
   * build G_munu from rectangles
   ********************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(rectangles)
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
  
    double U1[18];
    
    for ( int imunu = 0; imunu < 6; imunu++) {

      const int imu = dirpairs[imunu][0]; 
      const int inu = dirpairs[imunu][1]; 

      _cm_eq_zero ( U1 );

      _cm_pl_eq_cm ( U1, rectangles[g_iup[ix][imu]][imunu][0][0] );
      _cm_pl_eq_cm ( U1, rectangles[g_iup[ix][imu]][imunu][0][1] );
      _cm_pl_eq_cm ( U1, rectangles[g_idn[ix][imu]][imunu][0][0] );
      _cm_pl_eq_cm ( U1, rectangles[g_idn[ix][imu]][imunu][0][1] );

      _cm_pl_eq_cm ( U1, rectangles[g_iup[ix][inu]][imunu][1][0] );
      _cm_pl_eq_cm ( U1, rectangles[g_iup[ix][inu]][imunu][1][1] );
      _cm_pl_eq_cm ( U1, rectangles[g_idn[ix][inu]][imunu][1][0] );
      _cm_pl_eq_cm ( U1, rectangles[g_idn[ix][inu]][imunu][1][1] );

      _cm_eq_antiherm_cm ( Gr[ix][imunu], U1 );
      _cm_ti_eq_re ( Gr[ix][imunu], one_over_eight );
    }
  }
#if 0
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

      /* for O44 : G_{0,1} G_{1,0} + G_{0,2} G_{2,0} + G_{0,3} G_{3,0} 
       * indices        0       0         1       1         2       2  */

      for ( int nu = 0; nu < 3; nu++ ) {
        _cm_eq_cm_ti_cm ( s, G[ix][nu], G[ix][nu] );
        _re_pl_eq_tr_cm ( &(pl_tmp[0]), s );
      }

      /* for Okk : G_{1,2} G_{1,2} + G_{1,2} G_{1,2} + G_{1,2} G_{1,2} 
       * indices        3       3         4       4         5       5 */
      for ( int nu = 3; nu<6; nu++) {
        _cm_eq_cm_ti_cm ( s, G[ix][nu], G[ix][nu] );
        _re_pl_eq_tr_cm ( &(pl_tmp[1]), s );
      }
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

}  /* end of gluonic_operators_from_fst */


}  /* end of namespace cvc */
