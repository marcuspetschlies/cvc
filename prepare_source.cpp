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

namespace cvc {

int prepare_volume_source(double *s, unsigned int V) {

  int status = 0;
  double ratime, retime;

  ratime = _GET_TIME;

  switch(g_noise_type) {
    case 1:
      /* status = rangauss(s, 24*V); */
      status = rangauss(s, _GSI(V) );
      break;
    case 2:
      /* status = ranz2(s, 24*V); */
      status = ranz2(s, _GSI(V) );
      break;
  }

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [prepare_volume_source] time for prepare_volume_source = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );
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

  int i, exitstatus, x0, x1, x2, x3;
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
  exitstatus = MPI_Cart_rank(g_tr_comm, &i, &source_proc_id);
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

  unsigned int ix;
  int i, k, ia, isrc, ishift[4];
  int gcoords[4], lcoords[4], source_proc_id=0;
  int exitstatus;
  double norm[12];
  complex w;
  FILE*ofs=NULL;
  char filename[200];
  unsigned int *xid=NULL, *xid_count=NULL, xid_nc=0, **xid_member=NULL;
  double *xid_val = NULL;
  double **buffer=NULL;


  for(isrc=0; isrc<16; isrc++)
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

    if(source_proc_id == g_cart_id) {
      ix = g_lexic2eosub[g_ipt[lcoords[0]][lcoords[1]][lcoords[2]][lcoords[3]]];
    }
    for(ia=0; ia<12; ia++) {
      memset(sp[ia], 0, sizeof_field);
      if(source_proc_id == g_cart_id) {
        sp[ia][_GSI(ix)+2*ia] = 1.;
      }
    }

    exitstatus = project_propagator_field(sp[0], sp[0], 1, V[0], 12, numV, N);
    if(exitstatus != 0) {
      return(3);
    }

    for(ia=0; ia<12; ia++) {
      if(g_cart_id == source_proc_id) {
        ix = g_lexic2eosub[g_ipt[lcoords[0]][lcoords[1]][lcoords[2]][lcoords[3]]];
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

    for(ia=0; ia<12; ia++) {
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,w) shared(sp)
#endif
      for(ix=0; ix<N; ix++) {
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
    for(ia=0; ia<12; ia++) {
      for(i=0; i<xid_nc; i++) {
        for(k=0; k<xid_count[i]; k++) {
          buffer[ia][2*i  ] += sp[0][_GSI(xid_member[i][k])+2*ia  ];
          buffer[ia][2*i+1] += sp[0][_GSI(xid_member[i][k])+2*ia+1];
        }
      }
    }
    for(ia=0; ia<12; ia++) {
      for(i=0; i<xid_nc; i++) {
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
    for(i=0; i<xid_nc; i++) {
      for(ia=0; ia<12; ia++) {
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
  double ratime, retime;

  ratime = _GET_TIME;

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

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [init_clover_eo_spincolor_pointsource_propagator] time for init_clover_eo_spincolor_pointsource_propagator = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }
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

  int i, exitstatus, x0;

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
  if(g_proc_coords[3] == tseq / T ) {
#else
  if(g_proc_coords[0] == tseq / T ) {
#endif
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
  const size_t sizeof_spinor_field_timeslice = _GSI(LX*LY*LZ) * sizeof(double);

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


/**********************************************************
 * timeslice sources for one-end trick with spin dilution 
 **********************************************************/
int init_timeslice_source_oet(double **s, int tsrc, int*momentum, int init) {
  const unsigned int sf_items = _GSI(VOLUME);
  const size_t       sf_bytes = sf_items * sizeof(double);
  const int          have_source = ( tsrc / T == g_proc_coords[0] ) ? 1 : 0;
  const unsigned int VOL3 = LX*LY*LZ;

  unsigned int x1, x2, x3;
  static double *ran = NULL;
  double ratime, retime;
  
  ratime = _GET_TIME;

  if(init > 0) {
    if(ran != NULL) {
      free(ran);
      ran = NULL;
    }
  
    if(have_source) {
      fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d = (%d, %d, %d, %d) allocates random field\n", g_cart_id,
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
      ran = (double*)malloc(6*VOL3*sizeof(double));
      if(ran == NULL) {
        fprintf(stderr, "[init_timeslice_source_oet] Error from malloc\n");
        return(1);
      }
    }
  } else {
    if( have_source ) {
      if( ran == NULL ) {
        fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d Error, illegal call to function init_timeslice_source_oet with init = 0, random field is not initialized\n", g_cart_id);
        fflush(stdout);
        fprintf(stderr, "[init_timeslice_source_oet] proc%.4d Error, illegal call to function init_timeslice_source_oet with init = 0, random field is not initialized\n", g_cart_id);
        return(2);
      }
    }
  }  /* end of if init > 0 */

  /* initialize spinor fields to zero */
  memset(s[0], 0, sf_bytes);
  memset(s[1], 0, sf_bytes);
  memset(s[2], 0, sf_bytes);
  memset(s[3], 0, sf_bytes);

  if(have_source) {
    if(init > 0) {
      fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d drawing random vector\n", g_cart_id);

      switch(g_noise_type) {
        case 1:
          rangauss(ran, 6*VOL3);
          break;
        case 2:
          ranz2(ran, 6*VOL3);
          break;
      }
    } else {
      fprintf(stdout, "# [init_timeslice_source_oet] proc%.4d using existing random vector\n", g_cart_id);
    }

    const int timeslice = tsrc % T;  /*local timeslice */
    double *buffer = NULL;

    if(momentum != NULL) {
      const double       TWO_MPI = 2. * M_PI;
      const double       p[3] = {
         TWO_MPI * (double)momentum[0]/(double)LX_global,
         TWO_MPI * (double)momentum[1]/(double)LY_global,
         TWO_MPI * (double)momentum[2]/(double)LZ_global };
      const double phase_offset = p[0] * g_proc_coords[1] * LX + p[1] * g_proc_coords[2] * LY + p[2] * g_proc_coords[3] * LZ;
      buffer = (double*)malloc(6*VOL3*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
      unsigned int iix;
      double phase, cphase, sphase, tmp[6], *ptr;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(x1=0;x1<LX;x1++) {
      for(x2=0;x2<LY;x2++) {
      for(x3=0;x3<LZ;x3++) {
        phase = phase_offset + x1 * p[0] + x2 * p[1] + x3 * p[2];
        cphase = cos( phase );
        sphase = sin( phase );
        iix = 6 * g_ipt[0][x1][x2][x3];
        ptr = buffer + iix;
        memcpy(tmp, ran+iix, 6*sizeof(double));
        ptr[0] = tmp[0] * cphase - tmp[1] * sphase;
        ptr[1] = tmp[0] * sphase + tmp[1] * cphase;
        ptr[2] = tmp[2] * cphase - tmp[3] * sphase;
        ptr[3] = tmp[2] * sphase + tmp[3] * cphase;
        ptr[4] = tmp[4] * cphase - tmp[5] * sphase;
        ptr[5] = tmp[4] * sphase + tmp[5] * cphase;
      }}}
#ifdef HAVE_OPENMP
}
#endif
    } else {
      buffer = ran;
    } /* end of if momentum != NULL */


#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int ix, iix;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<VOL3; ix++) {

      iix = _GSI(timeslice * VOL3 + ix);

      /* set ith spin-component in ith spinor field */
      memcpy(s[0]+(iix + 6*0) , buffer+(6*ix), 6*sizeof(double) );
      memcpy(s[1]+(iix + 6*1) , buffer+(6*ix), 6*sizeof(double) );
      memcpy(s[2]+(iix + 6*2) , buffer+(6*ix), 6*sizeof(double) );
      memcpy(s[3]+(iix + 6*3) , buffer+(6*ix), 6*sizeof(double) );

    }  /* of ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
    if (momentum != NULL) free(buffer);
  }  /* end of if have source */

  if(init == 2) free(ran);

  retime = _GET_TIME;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [init_timeslice_source_oet] time for init_timeslice_source_oet = %e seconds\n", retime-ratime);
    fflush(stdout);
  }


  return(0);
}  /* end of init_timeslice_source_oet */


}  /* end of namespace cvc */
