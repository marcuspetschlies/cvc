/************************************************
 * prepare_source.cpp
 *
 * Di 12. Jul 17:02:38 CEST 2016
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <getopt.h>

#include "cvc_linalg.h"
#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "project.h"
#include "ranlxd.h"
#include "matrix_init.h"
#include "make_x_orbits.h"
#include "gamma.h"
#include "table_init_z.h"
#include "table_init_d.h"

namespace cvc {

int prepare_volume_source ( double * const s, unsigned int const V ) {

  int status = 0;

  switch(g_noise_type) {
    case 1:
      /* status = rangauss(s, 24*V); */
      status = rangauss(s, _GSI(V) );
      break;
    case 2:
      /* status = ranz2(s, 24*V); */
      status = ranz2(s, _GSI(V) );
      break;
    case 3:
      status = ranz3(s, _GSI(V/2) );
      break;
    default:
      if ( g_cart_id == 0 ) fprintf ( stderr, "[prepare_volume_source] Error, unknown noise type\n");
      status = 1;
      break;
  }

  return(status);
}  /* end of prepare_volume_source */

/*********************************************************
 * out: s_even, s_odd, even and odd part of source field
 * in: source_coords, global source coordinates
 *     have_source == 1 if process has source, otherwise 0
 *     work0, work1,... auxilliary eo work fields
 *
 * s_even and s_odd do not need to have halo sites
 * work0 should have halo sites
 *********************************************************/
int init_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, double*gauge_field, int sign, int have_source, double *work0) {
 
  unsigned int Vhalf = VOLUME/2;

  int local_source_coords[4] = { global_source_coords[0]%T, global_source_coords[1]%LX, global_source_coords[2]%LY, global_source_coords[3]%LZ };

  int source_location_iseven =  ( global_source_coords[0] + global_source_coords[1] + global_source_coords[2] + global_source_coords[3] ) % 2 == 0;

  double spinor1[24];
  size_t bytes = 24*Vhalf*sizeof(double);

  /* all procs: initialize to zero */
  memset(s_even, 0, bytes);
  memset(s_odd,  0, bytes);

  /* source node: set source */

  if(source_location_iseven) {
    if(have_source && g_verbose > 2 ) fprintf(stdout, "# [init_eo_spincolor_pointsource_propagator] even source location (%d,%d,%d,%d)\n",
        global_source_coords[0], global_source_coords[1], global_source_coords[2], global_source_coords[3]);

    /* source proces set point source */
    memset(work0,  0, bytes);
    if(have_source) {
      unsigned int eo_source_location = g_lexic2eosub[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ];
      work0[_GSI(eo_source_location) + 2*isc] = 1.0;
    }
    /* even component */
    /* xi_e = M^-1 eta_e */
    M_zz_inv (s_even, work0, sign*g_mu);

    /* odd component */
    /* xi_o = g5 X_oe M^-1 eta_e */
    memcpy(work0, s_even, bytes);
    xchange_eo_field(work0, 0);
    Hopping_eo(s_odd, work0, gauge_field, 1);
    g5_phi(s_odd, Vhalf);
    spinor_field_ti_eq_re(s_odd, -1., Vhalf);

  } else {
    if(have_source && g_verbose > 2 ) fprintf(stdout, "# [init_eo_spincolor_pointsource_propagator] odd source location (%d,%d,%d,%d)\n",
        global_source_coords[0], global_source_coords[1], global_source_coords[2], global_source_coords[3]);

    if(have_source) {
      unsigned int eo_source_location = g_lexic2eosub[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ];
      memset(spinor1, 0, 24*sizeof(double));
      spinor1[2*isc] = 1.0;
      _fv_eq_gamma_ti_fv( s_odd+_GSI( eo_source_location ), 5, spinor1 );
    }

  }

  return(0);
}  /* end of prepare_eo_spincolor_point_source */


/*********************************************************
 * finalize inversion (1 & X \\ 0 & 1) acting on C^-1 xi_o
 *
 * p_even, p_odd, r_even, r_odd do not need halo sites
 * work0 needs halo sites
 * p_even and r_even can be same memory region
 * p_odd  and r_odd  can be same memory region
 *********************************************************/
int fini_eo_propagator(double *p_even, double *p_odd, double *r_even, double *r_odd , double*gauge_field, int sign, double *work0) {
 
  const unsigned int Vhalf = VOLUME/2;
  const size_t bytes = 24*Vhalf*sizeof(double);

  /* work0 <- r_odd */
  memcpy( work0, r_odd, bytes);
  /* account for missing 1/2k in C / Cbar */
  /* work0 <- work0 x 2kappa = r_odd x 2kappa */
  spinor_field_ti_eq_re (work0, 2.*g_kappa, Vhalf);
  /* p_odd <- X_eo work0 = X_eo 2kappa r_odd; p_odd auxilliary field */
  X_eo (p_odd, work0, sign*g_mu, gauge_field);
  /* p_even <- p_odd + r_even = r_even + X_eo 2kappa r_odd */
  spinor_field_eq_spinor_field_pl_spinor_field(p_even, p_odd, r_even, Vhalf);
  /* p_odd <- work0 = 2kappa r_odd */
  memcpy(p_odd, work0, bytes);

  return(0);
}  /* end of fini_eo_propagator */

/***************************************************************************************************
 * prepare sequential source from propagator
 *
 * safe, if s_even = p_even or s_odd = p_odd
 ***************************************************************************************************/
int init_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, double*gauge_field, int sign, int pseq[3], int gseq, double *work0) {
  const unsigned int Vhalf = VOLUME/2;
  const unsigned int VOL3half = LX*LY*LZ/2;
  const int tloc = tseq % T;
  const size_t sizeof_eo_spinor_field = 24 * Vhalf * sizeof(double);
  const double MPI2 = 2. * M_PI;

  const double  q[3] = {MPI2 * pseq[0] / LX_global, MPI2 * pseq[1] / LY_global, MPI2 * pseq[2] / LZ_global};
  const double  q_offset = q[0] * g_proc_coords[1] * LX + q[1] * g_proc_coords[2] * LY + q[2] * g_proc_coords[3] * LZ;

  const size_t offset = _GSI(VOL3half);
  const size_t bytes  = offset * sizeof(double);

  int i, x0, x1, x2, x3;
  unsigned int ix;
  int source_proc_id = 0;
  double q_phase;
  double *s_=NULL, *p_=NULL, spinor1[24], spinor2[24];
  complex w;
  double ratime, retime;

  ratime = _GET_TIME;
#ifdef HAVE_MPI
  /* have seq source timeslice ? */
  i = tseq / T;
  int exitstatus = MPI_Cart_rank(g_tr_comm, &i, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[init_eo_sequential_source] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
  /* if(g_tr_id == source_proc_id) fprintf(stdout, "# [init_eo_sequential_source] proc %d / %d = (%d,%d,%d,%d) has t sequential %2d / %2d\n", g_cart_id, g_tr_id,
     g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], tseq, tloc); */
#endif

  if(g_tr_id == source_proc_id) {
#ifdef HAVE_OPENMP
#pragma omp parallel for private(x1,x2,x3,ix,q_phase,w,spinor1,spinor2,s_,p_)
#endif
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[tloc][x1][x2][x3];
      q_phase = q[0] * x1 + q[1] * x2 + q[2] * x3 + q_offset;
      w.re = cos(q_phase);
      w.im = sin(q_phase);
      if(g_iseven[ix]) {
        s_ = s_even + _GSI(g_lexic2eosub[ix]);
        p_ = p_even + _GSI(g_lexic2eosub[ix]);
      } else {
        s_ = s_odd + _GSI(g_lexic2eosub[ix]);
        p_ = p_odd + _GSI(g_lexic2eosub[ix]);
      }
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(spinor2, gseq, spinor1);
      // _fv_eq_fv(s_, spinor2);
      _fv_eq_gamma_ti_fv(s_, 5, spinor2);
    }}}
    for(i=1; i<T; i++) {
      x0 = (tloc + i) % T;
      memset(s_even+x0*offset, 0, bytes);
      memset(s_odd +x0*offset, 0, bytes);
    }
  } else {
    // printf("# [] process %d setting source to zero\n", g_cart_id);
    memset(s_even, 0, sizeof_eo_spinor_field);
    memset(s_odd,  0, sizeof_eo_spinor_field);
  }

  Q_eo_SchurDecomp_Ainv (s_even, s_odd, s_even, s_odd, gauge_field, sign*g_mu, work0);
  retime = _GET_TIME;

  if( g_cart_id == 0 ) fprintf(stdout, "# [init_eo_sequential_source] time for init_eo_sequential_source = %e seconds\n", retime-ratime);

  return(0);
}  /* end of init_eo_sequential_source */

/*********************************************************************************************************
 * sp - spinor fields without halo
 * V eigensystem
 * numV number of eigenvectors
 *********************************************************************************************************/

int check_vvdagger_locality (double** V, int numV, int base_coords[4], char*tag, double **sp) {

  const unsigned int N = T*LX*LY*LZ / 2;
  const size_t sizeof_field = _GSI(N) * sizeof(double);

  int ishift[4];
  int gcoords[4], lcoords[4], source_proc_id=0;
  int exitstatus;
  double norm[12];
  complex w;
  FILE*ofs=NULL;
  char filename[200];
  unsigned int *xid=NULL, *xid_count=NULL, xid_nc=0, **xid_member=NULL;
  double *xid_val = NULL;
  double **buffer=NULL;


  for ( int isrc=0; isrc<16; isrc++)
  {
    ishift[0] =  isrc      / 8;
    ishift[1] = (isrc % 8) / 4;
    ishift[2] = (isrc % 4) / 2;
    ishift[3] =  isrc % 2;

    gcoords[0] = ( base_coords[0] + ishift[0] * T_global /2 ) % T_global;
    gcoords[1] = ( base_coords[1] + ishift[1] * LX_global/2 ) % LX_global;
    gcoords[2] = ( base_coords[2] + ishift[2] * LY_global/2 ) % LY_global;
    gcoords[3] = ( base_coords[3] + ishift[3] * LZ_global/2 ) % LZ_global;

    if( ( gcoords[0] + gcoords[1] + gcoords[2] + gcoords[3]  )%2 == 0 ) {
      if(g_cart_id == 0) fprintf(stderr, "[check_vvdagger_locality] Warning, (%d, %d, %d, %d) is an even location, change by 1 in z-direction\n", 
            gcoords[0], gcoords[1], gcoords[2], gcoords[3]);
      gcoords[3] = ( gcoords[3] + 1 ) % LZ_global;
    }

    lcoords[0] = gcoords[0] % T;
    lcoords[1] = gcoords[1] % LX;
    lcoords[2] = gcoords[2] % LY;
    lcoords[3] = gcoords[3] % LZ;
#ifdef HAVE_MPI
    int source_proc_coords[4] =  { gcoords[0] / T, gcoords[1] / LX, gcoords[2] / LY, gcoords[3] / LZ};
    exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
    if(exitstatus != MPI_SUCCESS) {
      return(1);
    }
    if(source_proc_id == g_cart_id) {
      printf("# [check_vvdagger_locality] process %d has the source\n", g_cart_id);
    }
#endif

    for ( int ia=0; ia<12; ia++) {
      memset(sp[ia], 0, sizeof_field);
      if(source_proc_id == g_cart_id) {
        unsigned int const ix = g_lexic2eosub[g_ipt[lcoords[0]][lcoords[1]][lcoords[2]][lcoords[3]]];
        sp[ia][_GSI(ix)+2*ia] = 1.;
      }
    }

    exitstatus = project_propagator_field(sp[0], sp[0], 1, V[0], 12, numV, N);
    if(exitstatus != 0) {
      return(3);
    }

    for ( int ia=0; ia<12; ia++) {
      if(g_cart_id == source_proc_id) {
        unsigned int ix = g_lexic2eosub[g_ipt[lcoords[0]][lcoords[1]][lcoords[2]][lcoords[3]]];
        _co_eq_fv_dag_ti_fv(&w, sp[ia]+_GSI(ix), sp[ia]+_GSI(ix));
        norm[ia] = w.re;
      }
    }
#ifdef HAVE_MPI
    exitstatus = MPI_Bcast(norm, 12, MPI_DOUBLE, source_proc_id, g_cart_grid);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[check_vvdagger_locality] Error from MPI_Bcast, status was %d\n", exitstatus);
    }
#endif

    for ( int ia=0; ia<12; ia++) {
#ifdef HAVE_OPENMP
#pragma omp parallel for private(w) shared(sp)
#endif
      for ( unsigned int ix=0; ix<N; ix++) {
        _co_eq_fv_dag_ti_fv(&w, sp[ia]+_GSI(ix), sp[ia]+_GSI(ix) );
        sp[0][_GSI(ix)+2*ia  ] = w.re;
        sp[0][_GSI(ix)+2*ia+1] = w.im;
      }
    }


    exitstatus = init_x_orbits_4d(&xid, &xid_count, &xid_val, &xid_nc, &xid_member, gcoords);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[check_vvdagger_locality] Error from init_x_orbits_4d, status was %d\n", exitstatus);
      return(4);
    }
    fprintf(stdout, "# [check_vvdagger_locality] proc%.4d number of classes = %u\n", g_cart_id, xid_nc);

    init_2level_buffer(&buffer, 12, 2*xid_nc);
    memset(buffer[0], 0, 12*xid_nc*sizeof(double));
    for ( int ia=0; ia<12; ia++) {
      for ( unsigned int i=0; i<xid_nc; i++) {
        for ( unsigned int k=0; k<xid_count[i]; k++) {
          buffer[ia][2*i  ] += sp[0][_GSI(xid_member[i][k])+2*ia  ];
          buffer[ia][2*i+1] += sp[0][_GSI(xid_member[i][k])+2*ia+1];
        }
      }
    }
    for ( int ia=0; ia<12; ia++) {
      for ( unsigned int i=0; i<xid_nc; i++) {
        buffer[ia][2*i  ] /= xid_count[i];
        buffer[ia][2*i+1] /= xid_count[i];
      }
    }
  
    sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.proc%.4d", tag, numV, gcoords[0], gcoords[1], gcoords[2], gcoords[3], g_cart_id);
    ofs = fopen(filename, "w");
    if( ofs == NULL ) {
      fprintf(stderr, "[check_vvdagger_locality] Error from fopen\n");
      EXIT(2);
    }
    for ( unsigned int i=0; i<xid_nc; i++) {
      for ( int ia=0; ia<12; ia++) {
        fprintf(ofs, "%6d %16.7e %6u %3d %25.16e\n", i, xid_val[i], xid_count[i], ia, buffer[ia][2*i] );
      }
    }
    fclose(ofs);
#ifdef HAVE_MPI
    MPI_Barrier(g_cart_grid);
#endif
    fini_2level_buffer(&buffer);

    fini_x_orbits_4d(&xid, &xid_count, &xid_val, &xid_member);
  }  /* end of loop on shifted source locations */

  return(0);
}  /* end of check_vvdagger_locality */



/*********************************************************
 * out: s_even, s_odd, even and odd part of source field
 * in: source_coords, global source coordinates
 *     have_source == 1 if process has source, otherwise 0
 *     work0, work1,... auxilliary eo work fields
 *
 * - s_even and s_odd do not need to have halo sites
 *   work0 should have halo sites
 *
 * - for A^-1:
 *     mzzinv must ee
 *********************************************************/
int init_clover_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, double *gauge_field, double*mzzinv, int have_source, double *work0) {
 
  unsigned int Vhalf = VOLUME/2;

  int local_source_coords[4] = { global_source_coords[0]%T, global_source_coords[1]%LX, global_source_coords[2]%LY, global_source_coords[3]%LZ };

  int source_location_iseven =  (int)( ( global_source_coords[0] + global_source_coords[1] + global_source_coords[2] + global_source_coords[3] ) % 2 == 0 );

  double spinor1[24];
  size_t sizeof_eo_spinor_field = 24*Vhalf*sizeof(double);

  /* all procs: initialize to zero */
  memset(s_even, 0, sizeof_eo_spinor_field );
  memset(s_odd,  0, sizeof_eo_spinor_field );

  /* source node: set source */

  if(source_location_iseven) {

    if(have_source && g_verbose > 2 ) fprintf(stdout, "# [init_clover_eo_spincolor_pointsource_propagator] even source location (%d,%d,%d,%d)\n",
        global_source_coords[0], global_source_coords[1], global_source_coords[2], global_source_coords[3]);

    /* source proces set point source */
    memset(work0,  0, sizeof_eo_spinor_field );
    if(have_source) {
      unsigned int eo_source_location = g_lexic2eosub[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ];
      work0[_GSI(eo_source_location) + 2*isc] = 1.0;
    }
    /**********************************************
     * apply A^-1 g5
     * A^-1 = ( M_ee^-1 g5          0 )
     *        ( -g5 M_eo M_ee^-1 g5 1 )
     **********************************************/
    /* even component */
    /* xi_e = M^-1 eta_e */
    M_clover_zz_inv_matrix (s_even, work0, mzzinv);

    /* odd component */
    /* xi_o = g5 X_oe M^-1 eta_e */
    memcpy(work0, s_even, sizeof_eo_spinor_field );
    // xchange_eo_field(work0, 0);
    Hopping_eo(s_odd, work0, gauge_field, 1);
    g5_phi(s_odd, Vhalf);
    spinor_field_ti_eq_re(s_odd, -1., Vhalf);

  } else {
    /* odd source location */
    if(have_source && g_verbose > 2 ) fprintf(stdout, "# [init_clover_eo_spincolor_pointsource_propagator] odd source location (%d,%d,%d,%d)\n",
        global_source_coords[0], global_source_coords[1], global_source_coords[2], global_source_coords[3]);

    if(have_source) {
      unsigned int eo_source_location = g_lexic2eosub[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ];
      memset(spinor1, 0, 24*sizeof(double));
      spinor1[2*isc] = 1.0;
      _fv_eq_gamma_ti_fv( s_odd+_GSI( eo_source_location ), 5, spinor1 );
    }
  }
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  return(0);
}  /* end of prepare_clover_eo_spincolor_point_source */


/*********************************************************
 * finalize inversion (1 & X \\ 0 & 1) acting on C^-1 xi_o
 *
 * p_even, p_odd, r_even, r_odd do not need halo sites
 * work0 needs halo sites
 * p_even and r_even can be same memory region
 * p_odd  and r_odd  can be same memory region
 *********************************************************/
int fini_clover_eo_propagator(double *p_even, double *p_odd, double *r_even, double *r_odd , double*gauge_field, double*mzzinv, double *work0) {
 
  const unsigned int Vhalf = VOLUME/2;
  const size_t bytes = 24*Vhalf*sizeof(double);

  /* work0 <- r_odd */
  memcpy( work0, r_odd, bytes);
  /* account for missing 1/2k in C / Cbar */
  /* work0 <- work0 x 2kappa = r_odd x 2kappa */
  spinor_field_ti_eq_re (work0, 2.*g_kappa, Vhalf);
  /* p_odd <- X_eo work0 = X_eo 2kappa r_odd; p_odd auxilliary field */
  X_clover_eo (p_odd, work0, gauge_field, mzzinv);

  /* p_even <- p_odd + r_even = r_even + X_eo 2kappa r_odd */
  spinor_field_eq_spinor_field_pl_spinor_field(p_even, p_odd, r_even, Vhalf);
  /* p_odd <- work0 = 2kappa r_odd */
  memcpy(p_odd, work0, bytes);

  return(0);
}  /* end of fini_clover_eo_propagator */

/***************************************************************************************************
 * prepare sequential source from propagator
 *
 * safe, if s_even = p_even or s_odd = p_odd
 ***************************************************************************************************/
int init_clover_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, double*gauge_field, double*mzzinv, int pseq[3], int gseq, double *work0) {
  const unsigned int Vhalf = VOLUME/2;
  const unsigned int VOL3half = LX*LY*LZ/2;
  const int tloc = tseq % T;
  const size_t sizeof_eo_spinor_field = 24 * Vhalf * sizeof(double);
  const double MPI2 = 2. * M_PI;

  const double  q[3] = {MPI2 * pseq[0] / LX_global, MPI2 * pseq[1] / LY_global, MPI2 * pseq[2] / LZ_global};
  const double  q_offset = q[0] * g_proc_coords[1] * LX + q[1] * g_proc_coords[2] * LY + q[2] * g_proc_coords[3] * LZ;

  const size_t offset = _GSI(VOL3half);
  const size_t sizeof_eo_spinor_field_timeslice  = offset * sizeof(double);

  int i, x0;

  /* have seq source timeslice ? */
  const int source_proc_id = ( g_proc_coords[0] == tseq / T ) ? g_cart_id : -1;

  if(g_cart_id == source_proc_id && g_verbose > 2) fprintf(stdout, "# [init_clover_eo_sequential_source] proc %d = (%d,%d,%d,%d) has t sequential %2d / %2d\n", g_cart_id,
     g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], tseq, tloc);


  if(g_cart_id == source_proc_id) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int ix, ixeosub;
    double q_phase;
    double *s_=NULL, *p_=NULL, spinor1[24];
    complex w;
    /*************/
    /* even part */
    /*************/
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<VOL3half; ix++) {
      ixeosub  = tloc * VOL3half + ix;
      q_phase = q[0] * g_eosubt2coords[0][tloc][ix][0] + q[1] * g_eosubt2coords[0][tloc][ix][1] + q[2] * g_eosubt2coords[0][tloc][ix][2] + q_offset;
      /* fprintf(stdout, "# [init_clover_eo_sequential_source] proc%.4d t = %3d x_e = %6u = %3d %3d %3d\n", g_cart_id, tloc, ixeosub, 
          g_eosubt2coords[0][tloc][ix][0], g_eosubt2coords[0][tloc][ix][1], g_eosubt2coords[0][tloc][ix][2]); */
      w.re = cos(q_phase);
      w.im = sin(q_phase);
      s_ = s_even + _GSI(ixeosub);
      p_ = p_even + _GSI(ixeosub);
      
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(s_, gseq, spinor1);
      _fv_ti_eq_g5(s_);
    }  /* end of loop on ix */


    /*************/
    /* odd part  */
    /*************/
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<VOL3half; ix++) {
      ixeosub  = tloc * VOL3half + ix;
      q_phase = q[0] * g_eosubt2coords[1][tloc][ix][0] + q[1] * g_eosubt2coords[1][tloc][ix][1] + q[2] * g_eosubt2coords[1][tloc][ix][2] + q_offset;
      /* fprintf(stdout, "# [init_clover_eo_sequential_source] proc%.4d t = %3d x_o = %6u = %3d %3d %3d\n", g_cart_id, tloc, ixeosub,
          g_eosubt2coords[1][tloc][ix][0], g_eosubt2coords[1][tloc][ix][1], g_eosubt2coords[1][tloc][ix][2]); */
      w.re = cos(q_phase);
      w.im = sin(q_phase);
      s_ = s_odd + _GSI(ixeosub);
      p_ = p_odd + _GSI(ixeosub);
      
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(s_, gseq, spinor1);
      _fv_ti_eq_g5(s_);
    }  /* end of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    /***************************************/
    /* remaining timeslice are set to zero */
    /***************************************/
    for(i=1; i<T; i++) {
      x0 = (tloc + i) % T;
      memset(s_even+x0*offset, 0, sizeof_eo_spinor_field_timeslice );
      memset(s_odd +x0*offset, 0, sizeof_eo_spinor_field_timeslice );
    }
  } else {
    // printf("# [] process %d setting source to zero\n", g_cart_id);
    memset(s_even, 0, sizeof_eo_spinor_field);
    memset(s_odd,  0, sizeof_eo_spinor_field);
  }

  Q_clover_eo_SchurDecomp_Ainv (s_even, s_odd, s_even, s_odd, gauge_field, mzzinv, work0);

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
  return(0);
}  /* end of init_clover_eo_sequential_source */


/*************************************************************************
 * prepare a sequential source
 *************************************************************************/
int init_sequential_source(double *s, double *p, int tseq, int pseq[3], int gseq) {

  const double px = 2. * M_PI * pseq[0] / (double)LX_global;
  const double py = 2. * M_PI * pseq[1] / (double)LY_global;
  const double pz = 2. * M_PI * pseq[2] / (double)LZ_global;
  const double phase_offset =  g_proc_coords[1]*LX * px + g_proc_coords[2]*LY * py + g_proc_coords[3]*LZ * pz;
  const size_t sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

  int have_source=0, lts=-1;

  memset(s, 0, sizeof_spinor_field);

  if(s == NULL || p == NULL) {
    fprintf(stderr, "[init_sequential_source] Error, field is null\n");
    return(1);
  }

  /* (0) which processes have source? */
#if ( (defined PARALLELTX) || (defined PARALLELTXY) ) && (defined HAVE_QUDA)
  if(g_proc_coords[3] == tseq / T )
#else
  if(g_proc_coords[0] == tseq / T )
#endif
  {
    have_source = 1;
    lts = tseq % T;
  } else {
    have_source = 0;
    lts = -1;
  }
  if(have_source) {
    fprintf(stdout, "# [init_sequential_source] process %d has source\n", g_cart_id);
    fprintf(stdout, "# [init_sequential_source] t = %2d gamma id = %d p = (%d, %d, %d)\n", tseq, gseq, pseq[0], pseq[1], pseq[2]);
  }

  /* (1) multiply with phase and Gamma structure */
  if(have_source) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    double spinor1[24], phase;
    unsigned int ix;
    int x1, x2, x3;
    complex w;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(x1=0;x1<LX;x1++) {
    for(x2=0;x2<LY;x2++) {
    for(x3=0;x3<LZ;x3++) {
      ix = _GSI(g_ipt[lts][x1][x2][x3]);
      phase = phase_offset + x1 * px + x2 * py + x3 * pz;
      w.re =  cos(phase);
      w.im =  sin(phase);
      _fv_eq_gamma_ti_fv(spinor1, gseq, p + ix);
      _fv_eq_fv_ti_co(s + ix, spinor1, &w);
    }}}
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  }  /* of if have_source */

  return(0);
}  /* end of function init_sequential_source */


/*************************************************************************
 * prepare a coherent sequential source
 *************************************************************************/
int init_coherent_sequential_source(double *s, double **p, int tseq, int ncoh, int pseq[3], int gseq) {

  const double px = 2. * M_PI * pseq[0] / (double)LX_global;
  const double py = 2. * M_PI * pseq[1] / (double)LY_global;
  const double pz = 2. * M_PI * pseq[2] / (double)LZ_global;
  const double phase_offset =  g_proc_coords[1]*LX * px + g_proc_coords[2]*LY * py + g_proc_coords[3]*LZ * pz;
  const size_t sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

  int have_source=0, lts=-1, icoh;

  if(s == NULL || p == NULL) {
    fprintf(stderr, "[init_coherent_sequential_source] Error, field is null\n");
    return(1);
  }

  /* initalize target field s to zero */
  memset(s, 0, sizeof_spinor_field);

  /* loop on coherent source timeslices */
  for(icoh=0; icoh<ncoh; icoh++) {

    int tcoh = ( tseq + (T_global / ncoh) * icoh ) % T_global;

    /* (0) which processes have source? */
#if ( (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) ) && (defined HAVE_QUDA)
    if(g_proc_coords[3] == tcoh / T )
#else
    if(g_proc_coords[0] == tcoh / T )
#endif
    {
      have_source = 1;
      lts = tcoh % T;
    } else {
      have_source = 0;
      lts = -1;
    }
    if(have_source) {
      fprintf(stdout, "# [init_coherent_sequential_source] process %d has source using t = %2d gamma id = %2d p = (%3d, %3d, %3d)\n",
          g_cart_id, tcoh, gseq, pseq[0], pseq[1], pseq[2]);
    }

    /* (1) multiply with phase and Gamma structure */
    if(have_source) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
      double spinor1[24], phase;
      unsigned int ix;
      int x1, x2, x3;
      complex w;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        ix = _GSI(g_ipt[lts][x1][x2][x3]);
        phase = phase_offset + x1 * px + x2 * py + x3 * pz;
        w.re =  cos(phase);
        w.im =  sin(phase);
        _fv_eq_gamma_ti_fv(spinor1, gseq, p[icoh] + ix);
        _fv_eq_fv_ti_co(s + ix, spinor1, &w);
      }}}
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    }  /* of if have_source */

  }  /* end of loop on coherent sources */

  return(0);
}  /* end of function init_sequential_source */

/**********************************************************/
/**********************************************************/

/**********************************************************
 * timeslice sources for one-end trick with spin dilution 
 *
 * spin_dilution  = 1 or 4 ( no dilution, full dilution )
 * color_dilution = 1 or 3 ( no dilution, full dilution )
 **********************************************************/
int init_timeslice_source_oet ( double ** const s, int const tsrc, int * const momentum, int const spin_dilution, int const color_dilution, int const init) {

  const unsigned int sf_items = _GSI(VOLUME);
  const size_t       sf_bytes = sf_items * sizeof(double);
  const int          have_source = ( tsrc / T == g_proc_coords[0] ) ? 1 : 0;
  const unsigned int VOL3 = LX*LY*LZ;
  const int nspin = ( spin_dilution  == 0 ) ? 0 : 4 / spin_dilution;
  const int ncol  = ( color_dilution == 0 ) ? 0 : 3 / color_dilution;
  const int nsc   = nspin * ncol;

  static double *ran = NULL;
  double ratime, retime;
  
  ratime = _GET_TIME;

  if(init > 0) {
    if(ran != NULL) {
      free(ran);
      ran = NULL;
    }
  
    if(have_source) {
      if ( g_verbose > 0 ) fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d = (%d, %d, %d, %d) allocates random field\n", g_cart_id,
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

      ran = (double*)malloc( 2 * nsc * VOL3*sizeof(double));
      if(ran == NULL) {
        fprintf(stderr, "[init_timeslice_source_oet] Error from malloc\n");
        return(1);
      }
    }
  } else if ( init == 0 ) {
    if( have_source ) {
      if( ran == NULL ) {
        fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d Error, illegal call to function init_timeslice_source_oet with init = 0, random field is not initialized\n",
            g_cart_id);
        fflush(stdout);
        fprintf(stderr, "[init_timeslice_source_oet] proc%.4d Error, illegal call to function init_timeslice_source_oet with init = 0, random field is not initialized\n",
            g_cart_id);
        return(2);
      }
    }
  } else if ( init == -1 ) {
    /**********************************************************
     * recover ran from existing set of sources
     **********************************************************/
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [init_timeslice_source_oet] recovering random field from stochastic source\n");
    }
    if ( ran != NULL ) { free(ran); ran = NULL; }
    if ( have_source ) {
        const int timeslice = tsrc % T;  /*local timeslice */
        ran = ( double * )malloc ( 2 * nsc * VOL3 * sizeof(double));
        if(ran == NULL) {
          fprintf(stderr, "[init_timeslice_source_oet] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(1);
        }
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned ix=0; ix<VOL3; ix++) {
        unsigned int const iix = _GSI(timeslice * VOL3 + ix);
  
        for ( int ispin = 0; ispin < nspin; ispin++ ) {
          for ( int icol = 0; icol < ncol; icol++ ) {
            ran[2*( nsc * ix + ispin*ncol+icol)  ] = s[0][iix + 2*(3*ispin+icol)  ];
            ran[2*( nsc * ix + ispin*ncol+icol)+1] = s[0][iix + 2*(3*ispin+icol)+1];
          }
        }
      }
    }  /* end of if have_source */

    return(0);

  } else if ( init == -2 ) {
    /* free ran field an return  */
    if ( g_verbose > 2 && g_cart_id == 0 ) fprintf ( stdout, "# [init_timeslice_source_oet] fini ran field\n");
    if ( ran != NULL ) free ( ran );
    ran = NULL;
 
    return ( 0 );
 
  }  /* end of if init > 0 */

  /* initialize spinor fields to zero */
  for ( int i = 0; i < spin_dilution * color_dilution; i++ ) {
    memset(s[i], 0, sf_bytes);
  }

  if(init > 0) {
    if ( g_verbose > 0 && have_source ) fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d drawing random vector\n", g_cart_id);

    unsigned int const gVOL3 = LX_global * LY_global * LZ_global;
    unsigned int const items = 2 * nsc * gVOL3;
    double *ran_buffer = init_1level_dtable ( items );
    if ( ran_buffer == NULL ) {
      fprintf ( stderr, "[init_timeslice_source_oet] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      return ( 1 );
    }

    switch(g_noise_type) {
      case 1:
        rangauss( ran_buffer, items );
        break;
      case 2:
        ranz2 ( ran_buffer, items );
        break;
      case 4:
        ranbinary ( ran_buffer, items );
        break;
    }

    /* for ( unsigned int ix = 0; ix < 2*nsc*gVOL3; ix++ ) {
      fprintf ( stdout, " proc %4d x %6d ran_buffer %25.16e\n", g_cart_id, ix, ran_buffer[ix] );
    } */

    if ( have_source ) {

      /**********************************************************
       * source process copy your part from the global timeslice
       * field
       **********************************************************/
      if ( g_verbose > 1 ) fprintf ( stdout, "# [init_timeslice_source_oet] proc %d = %3d %3d %3d %3d copy timeslice data\n", g_cart_id, 
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3] );
      // memcpy ( ran, ran_buffer[tsrc], 6*VOL3 );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for ( int x1 = 0; x1 < LX; x1++ ) {
        int const y1 = x1 + g_proc_coords[1] * LX;
      for ( int x2 = 0; x2 < LY; x2++ ) {
        int const y2 = x2 + g_proc_coords[2] * LY;
      for ( int x3 = 0; x3 < LZ; x3++ ) {
        int const y3 = x3 + g_proc_coords[3] * LZ;

        /* global 3-dim index */
        unsigned int const iy = ( y1 * LY_global + y2 ) * LZ_global + y3;
        /* local 3-dim index */
        unsigned int const ix = ( x1 * LY        + x2 ) * LZ        + x3;

        double * const b_ = ran_buffer + 2*nsc*iy;
        double * const r_ = ran        + 2*nsc*ix;
        memcpy ( r_, b_, 2*nsc*sizeof ( double ) );
      }}}
    }  /* end of if have_source */

    fini_1level_dtable ( &ran_buffer );

  } else {
    fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d using existing random vector\n", g_cart_id);
  }

  if(have_source) {
    const int timeslice = tsrc % T;  /*local timeslice */
    double *buffer = NULL;

    if(momentum != NULL) {
      /* multiply the momentum phase to ran, save into buffer */
      const double       TWO_MPI = 2. * M_PI;
      const double       p[3] = {
         TWO_MPI * (double)momentum[0]/(double)LX_global,
         TWO_MPI * (double)momentum[1]/(double)LY_global,
         TWO_MPI * (double)momentum[2]/(double)LZ_global };
      const double phase_offset = p[0] * g_proc_coords[1] * LX + p[1] * g_proc_coords[2] * LY + p[2] * g_proc_coords[3] * LZ;
      buffer = (double*)malloc(2*nsc*VOL3*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int x1=0; x1<LX; x1++) {
      for ( unsigned int x2=0; x2<LY; x2++) {
      for ( unsigned int x3=0; x3<LZ; x3++) {
        double const phase = phase_offset + x1 * p[0] + x2 * p[1] + x3 * p[2];
        double const cphase = cos( phase );
        double const sphase = sin( phase );
        unsigned int const iix = 2 * nsc * g_ipt[0][x1][x2][x3];
        double * const ptr = buffer + iix;

        double tmp[2];

        for ( int k = 0; k < nsc; k++ ) {
          double const tmp[2] = { ran[iix + 2*k], ran[iix + 2*k+1] };

          ptr[2*k  ] = tmp[0] * cphase - tmp[1] * sphase;
          ptr[2*k+1] = tmp[0] * sphase + tmp[1] * cphase;
 
        }  /* end of loop on non-diluted spin-color components */  
  
      }}}
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
    } else {
      /* nothing to be done, just set buffer <- ran */
      buffer = ran;
    } /* end of if momentum != NULL */


#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix=0; ix<VOL3; ix++) {

      unsigned int const iix = _GSI(timeslice * VOL3 + ix);

      /* set ith spin-component in ith spinor field */
      int isc = 0;
      for ( int ispin = 0; ispin < spin_dilution; ispin++ ) {
        for ( int icol = 0; icol < color_dilution; icol++ ) {

          for ( int j = 0; j < nspin; j++ ) {
            for ( int k = 0; k < ncol; k++ ) {
              s[isc][ iix + 2 * ( 3 * (ispin*nspin+j)+(icol*ncol+k) )  ] = buffer[2 * ( nsc * ix + ncol * j + k )  ];
              s[isc][ iix + 2 * ( 3 * (ispin*nspin+j)+(icol*ncol+k) )+1] = buffer[2 * ( nsc * ix + ncol * j + k )+1];
     
          }}  /* end of loop on non-diluted spin-color indices */
          isc++;
      }}  /* end of loop on diluted spin-color indices */
    }  /* of ix */

    if ( momentum != NULL ) free(buffer);

  }  /* end of if have source */

  if(init == 2) free(ran);

  retime = _GET_TIME;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [init_timeslice_source_oet] time for init_timeslice_source_oet = %e seconds\n", retime-ratime);
    fflush(stdout);
  }

  return(0);
}  /* end of init_timeslice_source_oet */

/**********************************************************/
/**********************************************************/

/**********************************************************
 * oet Z3 timeslice sources with spin-color dilution 
 **********************************************************/
int init_timeslice_source_z3_oet ( double ** const s, int const  tsrc, int const momentum[3], int const init ) {
  const unsigned int sf_items = _GSI(VOLUME);
  const size_t       sf_bytes = sf_items * sizeof(double);
  const int          have_source = ( tsrc / T == g_proc_coords[0] ) ? 1 : 0;
  const unsigned int VOL3 = LX*LY*LZ;

  static double *ran = NULL;
  
  if(init > 0) {
    if(ran != NULL) {
      free(ran);
      ran = NULL;
    }
  
    if(have_source) {
      if ( g_verbose > 1 ) fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d = (%d, %d, %d, %d) allocates random field\n", g_cart_id,
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
      ran = ( double * ) malloc( 2 * VOL3 * sizeof(double) );
      if(ran == NULL) {
        fprintf(stderr, "[init_timeslice_source_z3_oet] Error from malloc %s %d\n", __FILE__, __LINE__ );
        return(1);
      }
    }
  } else if ( init == 0 ) {
    if( have_source ) {
      if( ran == NULL ) {
        fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d Error, illegal call to function init_timeslice_source_z3_oet with init = 0, random field is not initialized\n", g_cart_id);
        fflush(stdout);
        fprintf(stderr, "[init_timeslice_source_z3_oet] proc%.4d Error, illegal call to function init_timeslice_source_z3_oet with init = 0, random field is not initialized\n", g_cart_id);
        return(2);
      }
    }
  } else if ( init == -1 ) {
    /**********************************************************
     * recover ran from existing set of sources
     * copy the spin-color component number 0 into the ran
     * field for each site in the timeslice
     **********************************************************/
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [init_timeslice_source_z3_oet] recovering random field from stochastic source\n");
    }
    if ( ran != NULL ) { free(ran); ran = NULL; }
    if ( have_source ) {
        const int timeslice = tsrc % T;  /* local timeslice */
        ran = (double*)malloc( 2 * VOL3 * sizeof(double) );
        if(ran == NULL) {
          fprintf(stderr, "[init_timeslice_source_z3_oet] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(1);
        }
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned ix=0; ix<VOL3; ix++) {
        unsigned int const iix = _GSI(timeslice * VOL3 + ix);

        ran[2*ix  ] = s[0][iix  ];
        ran[2*ix+1] = s[0][iix+1];
      }
    }
    return(0);
  }  /* end of if init > 0 */

  /* initialize spinor fields to zero */

  if ( init >= 0 ) {
    /**********************************************************
     * whichever non-negative value for init, set
     * spinor fields to zero
     **********************************************************/
    for ( int i = 0; i < 12; i++ ) {
      memset ( s[i], 0, sf_bytes );
    }
  }

  if ( have_source && init >= 0 ) {
    /**********************************************************
     * Z_3 sampling
     **********************************************************/
    if ( init > 0 ) {
      if ( g_verbose > 0 ) fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d drawing random vector\n", g_cart_id);
      ranz3 ( ran, VOL3 );

    } else {
      if ( g_verbose > 0 ) fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d using existing random vector\n", g_cart_id);
    }

    const int timeslice = tsrc % T;  /*local timeslice */
    double *buffer = NULL;

    if(momentum != NULL) {
      /**********************************************************
       * add momentum factor
       **********************************************************/
      const double       TWO_MPI = 2. * M_PI;
      const double       p[3] = {
         TWO_MPI * (double)momentum[0]/(double)LX_global,
         TWO_MPI * (double)momentum[1]/(double)LY_global,
         TWO_MPI * (double)momentum[2]/(double)LZ_global };
      const double phase_offset = p[0] * g_proc_coords[1] * LX + p[1] * g_proc_coords[2] * LY + p[2] * g_proc_coords[3] * LZ;
      buffer = ( double* ) malloc ( 2*VOL3*sizeof(double) );
      if( g_verbose > 4 ) {
        fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d p = %e %e %e\n", g_cart_id, p[0], p[1], p[2] );
        fprintf(stdout, "# [init_timeslice_source_z3_oet] proc%.4d phase offset = %e\n", g_cart_id, phase_offset );
      }
#ifdef HAVE_OPENMP
#pragma omp parallel shared(buffer, p, ran)
{
#endif
      double phase, cphase, sphase;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for( int x1 = 0; x1 < LX; x1++ ) {
      for( int x2 = 0; x2 < LY; x2++ ) {
      for( int x3 = 0; x3 < LZ; x3++ ) {
        phase = phase_offset + x1 * p[0] + x2 * p[1] + x3 * p[2];
        cphase = cos( phase );
        sphase = sin( phase );
        unsigned int const iix = 2 * g_ipt[0][x1][x2][x3];
        double * const b_ = buffer + iix;
        double * const r_ = ran    + iix;

        b_[0] = r_[0] * cphase - r_[1] * sphase;
        b_[1] = r_[0] * sphase + r_[1] * cphase;
      }}}
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
    } else {
      buffer = ran;
    } /* end of if momentum != NULL */


    for ( int isc = 0; isc < 12; isc++ ) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared (buffer,isc)
{
#pragma omp for
#endif
      for( unsigned int ix = 0; ix < VOL3; ix++ ) {

        unsigned int const iix = _GSI(timeslice * VOL3 + ix) + 2 * isc;

        /* set isc-th spin-component in ith spinor field */
        s[isc][iix  ] = buffer[2*ix  ];
        s[isc][iix+1] = buffer[2*ix+1];

      }  /* of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    }  /* end of loop on spin-color components */

    if (momentum != NULL) free(buffer);

  }  /* end of if have source */

  if ( init == 2 ) free(ran);
  if ( init == -2 ) {
    if ( ran != NULL ) free(ran);
    ran = NULL;
  }

  return(0);
}  /* end of init_timeslice_source_z3_oet */

/**********************************************************/
/**********************************************************/
#if 0
/**********************************************************
 * prepare sequential FHT source with loop
 **********************************************************/
int prepare_sequential_fht_loop_source ( double ** const seq_source, double _Complex *** const loop, double ** const prop, gamma_matrix_type * const gamma_mat, int const gamma_num, double _Complex * const ephase, int const type, gamma_matrix_type * const g5herm  ) {
  
  unsigned int const VOL3 = LX * LY * LZ;
#pragma omp parallel for
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

    double _Complex ** M = init_2level_ztable ( 12, 12 );
    double _Complex ** _loop = loop[ix];
    double _Complex ** _laux = init_2level_ztable ( 12, 12 );
    if ( g5herm == NULL ) {
      _scm_eq_scm(  _laux,  _loop );
    } else {
      _scm_eq_gamma_ti_scm_adj_ti_gamma ( _laux, g5herm->m, _loop,  g5herm->m );
    }

    for ( int ig = 0; ig < gamma_num; ig++ ) {
      _scm_pl_eq_gamma_ti_scm_ti_gamma ( M, gamma_mat[ig].m, _laux,  gamma_mat[ig].m );
    }  /* end of loop on vertex gamma list */

    fini_2level_ztable ( &_laux );

    unsigned int const ix3 = ix % VOL3;

    if ( type == 0 ) {

      /**********************************************************
       * ztmp = Tr [ M ]
       **********************************************************/
      double _Complex ztmp = 0.;

      _co_eq_tr_scm ( &ztmp, M );

      /**********************************************************
       * ztmp = ztmp * exp ( i p_seq x_seq )
       **********************************************************/
      ztmp *= ephase[ix3];

      /**********************************************************
       * seq_source = ztmp * prop = Tr[ gLg ] * exp(ipx) * prop
       **********************************************************/
      for ( int ib = 0; ib < 12; ib++ ) {
      for ( int ia = 0; ia < 12; ia++ ) {
        double _Complex a = ( prop[ib][_GSI(ix) + 2*ia] + I * prop[ib][_GSI(ix) + 2*ia+1] ) * ztmp;
        seq_source[ib][_GSI(ix) + 2*ia  ] = creal( a );
        seq_source[ib][_GSI(ix) + 2*ia+1] = cimag( a );
      }}
    } else if ( type == 1 ) {

      double _Complex U[12][12];

      for ( int ialpha = 0; ialpha < 4; ialpha++ ) {
      for ( int ia = 0; ia < 3; ia++ ) {
        int const ka = 3 * ialpha + ia;

        for ( int ibeta = 0; ibeta < 4; ibeta++ ) {
        for ( int ib = 0; ib < 3; ib++ ) {
          int const kb = 3 * ibeta + ib;

          U[ka][kb] = 0.;

          for ( int igamma = 0; igamma < 4; igamma++ ) {
          for ( int ic = 0; ic < 3; ic++ ) {
            int const kc = 3 * igamma + ic;
            U[ka][kb] += M[ka][kc] * ( prop[kb][_GSI(ix)+2*kc] + I * prop[kb][_GSI(ix)+2*kc+1] );
          }}

          U[ka][kb] *= ephase[ix3];
        }}
      }} 

      for( int ib = 0; ib  < 12; ib++ ) {
      for( int ia = 0; ia  < 12; ia++ ) {
        seq_source[ib][_GSI(ix)+2*ia  ] = creal( U[ia][ib] );
        seq_source[ib][_GSI(ix)+2*ia+1] = cimag( U[ia][ib] );
      }}
    }  /* end of if type */

    fini_2level_ztable ( &M );

  }  /* end of loop on volume */

  return (0);
}  /* end of prepare_sequential_fht_loop_source */
#endif

/**********************************************************/
/**********************************************************/

/**********************************************************
 * prepare sequential FHT source with loop
 **********************************************************/
int prepare_sequential_fht_loop_source ( double ** const seq_source, double _Complex *** const loop, double ** const prop, gamma_matrix_type * const gamma_mat, int const gamma_num, double _Complex * const ephase, int const type, gamma_matrix_type * const g5herm  ) {
  
  unsigned int const VOL3 = LX * LY * LZ;
#pragma omp parallel for
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

    double _Complex ** M = init_2level_ztable ( 12, 12 );
    double _Complex ** _loop = loop[ix];
    double _Complex ** _laux = init_2level_ztable ( 12, 12 );
    if ( g5herm == NULL ) {
      _scm_eq_scm(  _laux,  _loop );
    } else {
      _scm_eq_gamma_ti_scm_adj_ti_gamma ( _laux, g5herm->m, _loop,  g5herm->m );
    }


    unsigned int const ix3 = ix % VOL3;

    if ( type == 0 ) {
      /**********************************************************
       * build M <- sum_ig gamma[ig] x Tr [ L x gamma[ig] ]
       **********************************************************/
      for ( int ig = 0; ig < gamma_num; ig++ ) {

        /**********************************************************
         * ztmp = Tr [ L x g ]
         **********************************************************/
        double _Complex ztmp = 0.;
        _co_eq_tr_scm_ti_gamma( &ztmp, _laux, gamma_mat[ig].m );

        /**********************************************************
         * ztmp = ztmp * exp ( i p_seq x_seq )
         **********************************************************/
        ztmp *= ephase[ix3];

        for ( int ialpha = 0; ialpha < 4; ialpha++ ) {
        for ( int ibeta = 0; ibeta < 4; ibeta++ ) {
          for ( int ib = 0; ib < 3; ib++ ) {
            int const ka = 3 * ialpha + ib;
            int const kb = 3 * ibeta  + ib;

            M[ka][kb] -= gamma_mat[ig].m[ialpha][ibeta] * ztmp;
          }
        }}

      }  /* end of loop on gamma set */

    } else if ( type == 1 ) {
 
      /**********************************************************
       * build M <- sum_ig gamma[ig] x L x gamma[ig]
       **********************************************************/
      for ( int ig = 0; ig < gamma_num; ig++ ) {
        _scm_pl_eq_gamma_ti_scm_ti_gamma ( M, gamma_mat[ig].m, _laux,  gamma_mat[ig].m );
      }  /* end of loop on vertex gamma list */

    }  /* end of loop on type */

    /**********************************************************
     * build seq_source <- M x prop 
     **********************************************************/
    for ( int ibeta = 0; ibeta < 4; ibeta++ ) {
    for ( int ib = 0; ib < 3; ib++ ) {
      int const kb = 3 * ibeta + ib;

      for ( int ialpha = 0; ialpha < 4; ialpha++ ) {
      for ( int ia = 0; ia < 3; ia++ ) {
        int const ka = 3 * ialpha + ia;

        double _Complex ztmp = 0.;

        for ( int igamma = 0; igamma < 4; igamma++ ) {
        for ( int ic = 0; ic < 3; ic++ ) {
          int const kc = 3 * igamma + ic;
          ztmp += M[ka][kc] * ( prop[kb][_GSI(ix)+2*kc] + I * prop[kb][_GSI(ix)+2*kc+1] );
        }}

        seq_source[kb][_GSI(ix)+2*ka  ] = creal( ztmp );
        seq_source[kb][_GSI(ix)+2*ka+1] = cimag( ztmp );

      }}
    }} 

    fini_2level_ztable ( &M );
    fini_2level_ztable ( &_laux );

  }  /* end of loop on volume */

  return (0);
}  /* end of prepare_sequential_fht_loop_source */


/**********************************************************/
/**********************************************************/

/**********************************************************
 * prepare sequential FHT source with loop
 *
 * scalar_field is a complex = 2 x double per site field
 * here only the real part is used
 **********************************************************/
int prepare_sequential_fht_twinpeak_source ( double ** const seq_source, double ** const prop, double * const scalar_field, int const gamma_id, double _Complex * const ephase ) {

  unsigned int const VOL3 = LX * LY * LZ;

  for ( int isc = 0; isc < 12; isc++ ) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double spinor1[24];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      unsigned int const iix = _GSI( ix );
      double * const _p = prop[isc]       + iix;
      double * const _s = seq_source[isc] + iix;

      _fv_eq_gamma_ti_fv ( spinor1, gamma_id, _p );
      _fv_ti_eq_re ( spinor1, scalar_field[2*ix] );

      _fv_eq_fv_ti_d2 ( _s, spinor1, (double*)(ephase + (ix % VOL3)) );

    }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }

  return (0);
}  /* end of prepare_sequential_fht_twinpeak_source */
 
/**********************************************************/
/**********************************************************/

}  /* end of namespace cvc */
