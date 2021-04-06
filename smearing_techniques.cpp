/******************************************
 * smearing_techniques.cpp
 *
 * originally from
 *   smearing_techniques.cc
 *   Author: Marc Wagner
 *   Date: September 2007
 *
 * February 2010
 * ported to cvc_parallel, parallelized
 *
 * Fri Dec  9 09:28:18 CET 2016
 * ported to cvc
 ******************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#include <mpi.h>  
#endif
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "table_init_d.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"

namespace cvc {

/************************************************************
 * Performs a number of APE smearing steps
 *
 ************************************************************/
int APE_Smearing(double *smeared_gauge_field, double const APE_smearing_alpha, int const APE_smearing_niter) {

  if ( APE_smearing_niter <= 0 ) return(0);

#if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA )
  /***********************************************************
   * call  library function wrapper
   ***********************************************************/
  _performAPEnStep ( APE_smearing_niter, APE_smearing_alpha );

#else

  const unsigned int gf_items = 72*VOLUME;
  const size_t gf_bytes = gf_items * sizeof(double);

  int iter;
  double *smeared_gauge_field_old = NULL;
  alloc_gauge_field(&smeared_gauge_field_old, VOLUMEPLUSRAND);

  for(iter=0; iter<APE_smearing_niter; iter++) {

    memcpy((void*)smeared_gauge_field_old, (void*)smeared_gauge_field, gf_bytes);
#ifdef HAVE_MPI
    xchange_gauge_field(smeared_gauge_field_old);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int idx;
    double M1[18], M2[18];
    unsigned int index;
    unsigned int index_mx_1, index_mx_2, index_mx_3;
    unsigned int index_px_1, index_px_2, index_px_3;
    unsigned int index_my_1, index_my_2, index_my_3;
    unsigned int index_py_1, index_py_2, index_py_3;
    unsigned int index_mz_1, index_mz_2, index_mz_3;
    unsigned int index_pz_1, index_pz_2, index_pz_3;
    double *U = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(idx = 0; idx < VOLUME; idx++) {
  
      /************************
       * Links in x-direction.
       ************************/
      index = _GGI(idx, 1);
  
      index_my_1 = _GGI(g_idn[idx][2], 2);
      index_my_2 = _GGI(g_idn[idx][2], 1);
      index_my_3 = _GGI(g_idn[g_iup[idx][1]][2], 2);
  
      index_py_1 = _GGI(idx, 2);
      index_py_2 = _GGI(g_iup[idx][2], 1);
      index_py_3 = _GGI(g_iup[idx][1], 2);
  
      index_mz_1 = _GGI(g_idn[idx][3], 3);
      index_mz_2 = _GGI(g_idn[idx][3], 1);
      index_mz_3 = _GGI(g_idn[g_iup[idx][1]][3], 3);
  
      index_pz_1 = _GGI(idx, 3);
      index_pz_2 = _GGI(g_iup[idx][3], 1);
      index_pz_3 = _GGI(g_iup[idx][1], 3);
  
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative y-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
  
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive y-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
  
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative z-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
  
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive z-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
  
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
  
  
      /***********************
       * Links in y-direction.
       ***********************/
  
      index = _GGI(idx, 2);
  
      index_mx_1 = _GGI(g_idn[idx][1], 1);
      index_mx_2 = _GGI(g_idn[idx][1], 2);
      index_mx_3 = _GGI(g_idn[g_iup[idx][2]][1], 1);
  
      index_px_1 = _GGI(idx, 1);
      index_px_2 = _GGI(g_iup[idx][1], 2);
      index_px_3 = _GGI(g_iup[idx][2], 1);
  
      index_mz_1 = _GGI(g_idn[idx][3], 3);
      index_mz_2 = _GGI(g_idn[idx][3], 2);
      index_mz_3 = _GGI(g_idn[g_iup[idx][2]][3], 3);
  
      index_pz_1 = _GGI(idx, 3);
      index_pz_2 = _GGI(g_iup[idx][3], 2);
      index_pz_3 = _GGI(g_iup[idx][2], 3);
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative x-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive x-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative z-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive z-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
  
      /**************************
       * Links in z-direction.
       **************************/
  
      index = _GGI(idx, 3);
  
      index_mx_1 = _GGI(g_idn[idx][1], 1);
      index_mx_2 = _GGI(g_idn[idx][1], 3);
      index_mx_3 = _GGI(g_idn[g_iup[idx][3]][1], 1);
  
      index_px_1 = _GGI(idx, 1);
      index_px_2 = _GGI(g_iup[idx][1], 3);
      index_px_3 = _GGI(g_iup[idx][3], 1);
  
      index_my_1 = _GGI(g_idn[idx][2], 2);
      index_my_2 = _GGI(g_idn[idx][2], 3);
      index_my_3 = _GGI(g_idn[g_iup[idx][3]][2], 2);
  
      index_py_1 = _GGI(idx, 2);
      index_py_2 = _GGI(g_iup[idx][2], 3);
      index_py_3 = _GGI(g_iup[idx][3], 2);
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative x-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive x-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative y-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive y-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
    }  /* end of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on number of smearing steps */

  free(smeared_gauge_field_old);

#ifdef HAVE_MPI
  xchange_gauge_field(smeared_gauge_field);
#endif

#endif  /* of if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA ) */
  return(0);
}  /* end of APE_Smearing */


/*****************************************************
 *
 * Performs a number of Jacobi smearing steps
 *
 * smeared_gauge_field  = gauge field used for smearing (in)
 * psi = quark spinor (in/out)
 * N, kappa = Jacobi smearing parameters (in)
 *
 *****************************************************/
int Jacobi_Smearing(double *smeared_gauge_field, double *psi, int const N, double const kappa) {

  if ( N <= 0 ) return(0);

#if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA )
    _performWuppertalnStep ( psi, psi, N, kappa );
#else

  const unsigned int sf_items = _GSI(VOLUME);
  const size_t sf_bytes = sf_items * sizeof(double);
  const double norm = 1.0 / (1.0 + 6.0*kappa);

  int i1;
  double *psi_old = (double*)malloc( _GSI(VOLUME+RAND)*sizeof(double));
  if(psi_old == NULL) {
    fprintf(stderr, "[Jacobi_Smearing] Error from malloc\n");
    return(3);
  }

  if ( smeared_gauge_field == NULL ) {
    fprintf ( stderr, "[Jacobi_Smearing] Error, smeared_gauge_field is NULL\n");
    return (4);
  }
#ifdef HAVE_MPI
  xchange_gauge_field (smeared_gauge_field);
#endif

  /* loop on smearing steps */
  for(i1 = 0; i1 < N; i1++) {

    memcpy((void*)psi_old, (void*)psi, sf_bytes);
#ifdef HAVE_MPI
    xchange_field(psi_old);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif

    unsigned int idx, idy;
    unsigned int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
    unsigned int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
    double *s=NULL, spinor[24];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(idx = 0; idx < VOLUME; idx++) {
      index_s = _GSI(idx);

      index_s_mx = _GSI(g_idn[idx][1]);
      index_s_px = _GSI(g_iup[idx][1]);
      index_s_my = _GSI(g_idn[idx][2]);
      index_s_py = _GSI(g_iup[idx][2]);
      index_s_mz = _GSI(g_idn[idx][3]);
      index_s_pz = _GSI(g_iup[idx][3]);

      idy = idx;
      index_g_mx = _GGI(g_idn[idy][1], 1);
      index_g_px = _GGI(idy, 1);
      index_g_my = _GGI(g_idn[idy][2], 2);
      index_g_py = _GGI(idy, 2);
      index_g_mz = _GGI(g_idn[idy][3], 3);
      index_g_pz = _GGI(idy, 3);

      s = psi + _GSI(idy);
      _fv_eq_zero(s);

      /* negative x-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mx, psi_old + index_s_mx);
      _fv_pl_eq_fv(s, spinor);

      /* positive x-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_px, psi_old + index_s_px);
      _fv_pl_eq_fv(s, spinor);

      /* negative y-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_my, psi_old + index_s_my);
      _fv_pl_eq_fv(s, spinor);

      /* positive y-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_py, psi_old + index_s_py);
      _fv_pl_eq_fv(s, spinor);

      /* negative z-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mz, psi_old + index_s_mz);
      _fv_pl_eq_fv(s, spinor);

      /* positive z-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_pz, psi_old + index_s_pz);
      _fv_pl_eq_fv(s, spinor);


      /* Put everything together; normalization. */
      _fv_ti_eq_re(s, kappa);
      _fv_pl_eq_fv(s, psi_old + index_s);
      _fv_ti_eq_re(s, norm);
    }  /* end of loop on ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on smearing steps */

  free(psi_old);

#endif  /* of ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA ) */

  return(0);
}  /* end of Jacobi_Smearing */

/*****************************************************/
/*****************************************************/

inline double distance_square ( int * const r, int * const r0, int * const LL, int const N ) {

  double rr = 0.;

  for ( int i = 0; i < N; i++ ) {
    int L  = LL[i];
    int Lh = L/2;
    int s  = r[i] - r0[i];
    int s2 = ( s <= Lh && s > -Lh ) ? s : ( s > Lh/2 ? s - L : s + L);
    rr += (double)s2 * (double)s2;
  }
  return( rr );
}  /* end of distance_square */


/*****************************************************
 * rms radius of source
 *****************************************************/
int rms_radius ( double ** const r2, double ** const w2, double * const s, int const source_coords[4] ) {

  int LL[3] = { LX_global, LY_global, LZ_global };
  int dsrc[3] = { source_coords[1], source_coords[2], source_coords[3] };

  memset ( r2[0], 0, 4*T_global*sizeof(double) );
  memset ( w2[0], 0, 4*T_global*sizeof(double) );

  double ** r2_aux = init_2level_dtable ( T, 4 );
  double ** w2_aux = init_2level_dtable ( T, 4 );

  for ( int t = 0; t < T; t++ ) {

    double r2_accum[4] = {0., 0., 0., 0.};
    double w2_accum[4] = {0., 0., 0., 0.};

    for ( int x = 0; x < LX; x++ ) {
    for ( int y = 0; y < LY; y++ ) {
    for ( int z = 0; z < LZ; z++ ) {

      int d[3] = {
        x + g_proc_coords[1]*LX,
        y + g_proc_coords[2]*LY,
        z + g_proc_coords[3]*LZ };

      unsigned int const ix = _GSI(g_ipt[t][x][y][z]);

      double rr = distance_square ( d, dsrc, LL, 3 );

      for ( int ispin = 0; ispin < 4; ispin++ ) {

        double w = 0.;
        for ( int icol = 0; icol < 3; icol++ ) {
          double const a = s[ix+2*(3*ispin+icol)  ];
          double const b = s[ix+2*(3*ispin+icol)+1];
          w += a*a + b*b;
        }

        r2_accum[ispin] += rr * w;
        w2_accum[ispin] += w;

      }
    }}}
    r2_aux[t][0] = r2_accum[0];
    r2_aux[t][1] = r2_accum[1];
    r2_aux[t][2] = r2_accum[2];
    r2_aux[t][3] = r2_accum[3];
    w2_aux[t][0] = w2_accum[0];
    w2_aux[t][1] = w2_accum[1];
    w2_aux[t][2] = w2_accum[2];
    w2_aux[t][3] = w2_accum[3];
  }  /* end of loop on timeslices */

#ifdef HAVE_MPI
  double ** buffer = init_2level_dtable ( T, 4 );
  memset ( buffer[0], 0, 4*T*sizeof(double) );
  if ( MPI_Reduce( r2_aux[0], buffer[0], 4*T, MPI_DOUBLE, MPI_SUM, 0, g_ts_comm ) != MPI_SUCCESS) {
    fprintf(stderr, "[rms_radius] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  if ( MPI_Gather( buffer[0], 4*T, MPI_DOUBLE, r2[0], 4*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf(stderr, "[rms_radius] Error from MPI_Gather %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  if ( MPI_Reduce( w2_aux[0], buffer[0], 4*T, MPI_DOUBLE, MPI_SUM, 0, g_ts_comm ) != MPI_SUCCESS) {
    fprintf(stderr, "[rms_radius] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  if ( MPI_Gather( buffer[0], 4*T, MPI_DOUBLE, w2[0], 4*T, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
    fprintf(stderr, "[rms_radius] Error from MPI_Gather %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  fini_2level_dtable ( &buffer );
#else
  memcpy ( r2[0], r2_aux[0], 4*T*sizeof(double) );
  memcpy ( w2[0], w2_aux[0], 4*T*sizeof(double) );
#endif

  fini_2level_dtable ( &r2_aux );
  fini_2level_dtable ( &w2_aux );

  return(0);
}  /* end of rms_radius */

/*****************************************************
 *
 *****************************************************/
int source_profile ( double *s, int source_coords[4], char*prefix ) {
  
  int LL[3] = { LX_global, LY_global, LZ_global };

  int sx[4], source_proc_id, d[3];
  double norm = 0., rr;
  complex w;
  char filename[200];

  get_point_source_info ( source_coords, sx, &source_proc_id);

  if ( g_cart_id == source_proc_id ) {
    unsigned int ix = _GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]] );
    _co_eq_fv_dag_ti_fv ( &w, s+ix, s+ix  );
    norm = w.re;
    if ( g_verbose > 3 ) fprintf(stdout, "# [source_profile] proc%.4d gsx = %3d %3d %3d %3d norm = %16.7e\n", g_cart_id,
                             source_coords[0], source_coords[1], source_coords[2], source_coords[3], norm);
  }
#ifdef HAVE_MPI
  if ( MPI_Bcast ( &norm, 1, MPI_DOUBLE, source_proc_id, g_cart_grid ) != MPI_SUCCESS ) {
    fprintf(stderr, "[source_profile] Error from MPI_Bcast %s %d\n", __FILE__, __LINE__);
    return(1);
  }
#endif
  
  /*if ( source_coords[0] / T == g_proc_coords[0] )
    int t = source_coords[0] % T;*/
  for ( int t = 0; t < T; t++ )
  {

    int const tt = t + g_proc_coords[0] * T;

    /* write in partfile format */
    if ( prefix == NULL ) {
      sprintf( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.ts%d.proct%.2dprocx%.2dprocy%.2dprocz%.2d", "source_profile", Nconf, 
          source_coords[0], source_coords[1], source_coords[2], source_coords[3], tt,
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3] );
    } else {
      sprintf( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.ts%d.proct%.2dprocx%.2dprocy%.2dprocz%.2d", prefix, Nconf, 
          source_coords[0], source_coords[1], source_coords[2], source_coords[3], tt,
          g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3] );
    }
    FILE *ofs = fopen( filename, "w");
    if ( ofs == NULL ) {
      fprintf(stderr, "[source_profile] Error from fopen %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    for ( int x = 0; x < LX; x++ ) {
      d[0] = x + g_proc_coords[1]*LX;
      for ( int y = 0; y < LY; y++ ) {
        d[1] = y + g_proc_coords[2]*LY;
        for ( int z = 0; z < LZ; z++ ) {
          d[2] = z + g_proc_coords[3]*LZ;
          unsigned int ix = _GSI(g_ipt[t][x][y][z]);
          _co_eq_fv_dag_ti_fv ( &w, s+ix, s+ix  );

          rr = distance_square ( d, &(source_coords[1]), LL, 3 );

          /* fprintf(stdout, "# [rms_radius] proc%.4d x = %2d %2d %2d %2d rr = %25.16e w = %25.16e\n", g_cart_id,
           *           t+g_proc_coords[0]*T, x+g_proc_coords[1]*LX, y+g_proc_coords[2]*LY, z+g_proc_coords[3]*LZ, rr, w.re ); */

          fprintf(ofs, "%3d %3d %3d %8.0f %16.7e\n", d[0], d[1], d[2], rr, w.re/norm );
        }
      }
    }

    fclose ( ofs );
   }  /* end of if have source timeslice */
  return(0);
}  /* end of source_profile */

}  /* end of namespace cvc */
