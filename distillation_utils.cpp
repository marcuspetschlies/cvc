/***************************************************
 * distillation_utils.cpp
 ***************************************************/
 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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

#include "global.h"
#include "ilinalg.h"
#include "iblas.h"
#include "laplace_linalg.h"
#include "cvc_geometry.h"
#include "cvc_geometry_3d.h"
#include "mpi_init.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "ranlxd.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "matrix_init.h"
#include "distillation_utils.h"

namespace cvc {

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int distillation_prepare_source ( double *s, double**v, int evec_source, int spin_source, int timeslice ) {

  const unsigned int VOL3 = LX*LY*LZ;
  const int have_source = ( timeslice / T == g_proc_coords[0] );
  const size_t sizeof_colorvector = _GVI(1) * sizeof(double);

  double ratime = _GET_TIME;

  if ( have_source ) {
    if ( g_verbose > 3 ) fprintf(stdout, "# [distillation_prepare_source] proc%.4d %3d %3d %3d %3d has source %3d\n", g_cart_id, 
        g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], timeslice );

    const unsigned int offset = ( timeslice % T )* VOL3;

    for (unsigned int ix = 0; ix < VOL3; ix++ ) {
      unsigned int iix = offset + ix;
      memcpy ( s+_GSI(iix) + 6*spin_source, v[evec_source]+_GVI(ix), sizeof_colorvector );
    }

  }  /* end of if have source */

  double retime = _GET_TIME;
  if (g_cart_id == 0 ) fprintf ( stdout, "# [distillation_prepare_source] time for distillation_prepare_source = %e seconds\n", retime-ratime );
  return(0);
}  /* end of distillation_prepare_source */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int distillation_reduce_propagator ( double ***p, double *s, double ***v , int numV ) {

  const unsigned int VOL3 = LX*LY*LZ;
  int exitstatus;

  double _Complex Z_1 = 1.;
  double _Complex Z_0 = 0.;
  char CHAR_C = 'C', CHAR_N = 'N';
  int INT_K = 3*VOL3;
  int INT_M = numV;
  int INT_N = 4;

  double ratime = _GET_TIME;

  double ***r = NULL;
  exitstatus = init_3level_buffer ( &r, T, 4, _GVI(VOL3) );
  if ( exitstatus != 0 ) {
    fprintf( stderr, "[distillation_reduce_propagator] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }
  colorvector_field_from_spinor_field ( r[0][0], s ); 

  for ( int it = 0; it < T; it++ ) {

    double ttime = _GET_TIME;

    _F(zgemm) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(v[it][0]), &INT_K, (double _Complex*)(r[it][0]), &INT_K, &Z_0, (double _Complex*)(p[it][0]), &INT_M, 1, 1);

    ttime -= _GET_TIME;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [distillation_reduce_propagator] time for zgemm = %e seconds\n", -ttime );

#ifdef HAVE_MPI
    ttime = _GET_TIME;
    double *px = NULL;
    unsigned int items = numV * 4 * 2;
    exitstatus = init_1level_buffer ( &px, items );
    if ( exitstatus != 0 ) {
      fprintf( stderr, "[distillation_reduce_propagator] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }
    memcpy ( px, p[it][0], items* sizeof(double) );
    exitstatus = MPI_Allreduce( px, p[it][0], items, MPI_DOUBLE, MPI_SUM, g_ts_comm );
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[distillation_reduce_propagator] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(2);
    }
    fini_1level_buffer ( &px );
    ttime -= _GET_TIME;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [distillation_reduce_propagator] time for timeslice reduction = %e seconds\n", -ttime );
#endif

  }  /* end of loop on timeslice */

  fini_3level_buffer ( &r );

  double retime = _GET_TIME;

  if(g_cart_id == 0 ) fprintf ( stdout, "# [distillation_reduce_propagator] time for distillation_reduce_propagator = %e seconds\n", retime-ratime );
  return(0);

}  /* end of distillation_reduce_propagator */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int colorvector_field_from_spinor_field ( double *r, double *s  ) {

  const unsigned int VOL3 = LX*LY*LZ;
  const size_t sizeof_colorvector = _GVI(1) * sizeof(double);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( int it = 0; it < T; it++ ) {
    const unsigned int offset = it * VOL3;
    for ( int ispin = 0; ispin < 4; ispin++ ) {
      for ( unsigned int ix = 0; ix < VOL3; ix++ ) {
        memcpy ( r +  _GVI(1) * ( ( 4*it+ispin )*VOL3 + ix ), s + _GSI(offset+ix) + _GVI(1)*ispin , sizeof_colorvector );
      }
    }
  }
  return(0);
}  /* end of colorvector_field_from_spinor_field */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int distillation_write_perambulator (double ****p, int numV, struct AffWriter_s*affw, char *tag, int io_proc ) {

  int exitstatus;
  double _Complex **pVV = NULL;
  const size_t bytes = numV * sizeof(double _Complex );

  double ratime = _GET_TIME;

#ifdef HAVE_LHPC_AFF
  struct AffNode_s *affn = NULL, *affdir=NULL;

  if ( io_proc >= 1 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[distillation_write_perambulator] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }
#endif


  if ( io_proc >= 1 ) {
    exitstatus = init_2level_zbuffer ( &pVV, numV, numV );
    if(exitstatus != 0) {
      fprintf(stderr, "[distillation_write_perambulator] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(2);
    }

    for ( int it_snk = 0; it_snk < T; it_snk++ ) {
      int t_snk = ( it_snk + g_proc_coords[0] * T );

      for ( int s_snk = 0; s_snk < 4; s_snk++ ) {

        for ( int l_src = 0; l_src < numV; l_src++ ) {
          memcpy ( pVV[l_src], p[l_src][it_snk][s_snk], bytes );
        }

#ifdef HAVE_LHPC_AFF
        char aff_buffer_path[200];
        sprintf( aff_buffer_path, "%s/tsnk%.2d/ssnk%d", tag, t_snk, s_snk );
        /* fprintf(stdout, "# [distillation_write_perambulator] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        exitstatus = aff_node_put_complex (affw, affdir, pVV[0], (uint32_t)numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[distillation_write_perambulator] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }
#endif

      }  /* end of loop on spin at sink */
    }  /* end of loop on sink timeslice */

    fini_2level_zbuffer ( &pVV );

  }  /* end of if io_proc >= 1 */ 

  double retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf ( stdout, "# [distillation_write_perambulator] time for distillation_write_perambulator = %e seconds\n", retime-ratime );

  return(0);
}  /* end of distillation_write_perambulator */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * color vector field norm diff
 ***********************************************************/

void colorvector_field_norm_diff_timeslice (double*d, double *r, double *s, int it, unsigned int N ) {

  int exitstatus;
  double daccum=0.;
  const unsigned int offset = it * N;
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) shared(d,r,s,N,daccum)
{
#endif
  double v1[_GVI(1)];
  double daccumt = 0.;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) {
    unsigned int iix = _GVI(offset + ix);
    _cv_eq_cv_mi_cv( v1, r+iix, s+iix);
    _re_pl_eq_cv_dag_ti_cv( daccumt,v1,v1);
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  daccum += daccumt;

  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#else
  daccum = daccumt;
#endif

#ifdef HAVE_MPI
  double daccumx = 0.;
  exitstatus = MPI_Allreduce(&daccum, &daccumx, 1, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if ( exitstatus != 0 ) fprintf ( stderr, "[colorvector_field_norm_diff_timeslice] Error from MPI_Allreduce, status was %d\n" );

  *d = sqrt(daccumx);
#else
  *d = sqrt( daccum );
#endif
}  /* end of colorvector_field_norm_diff */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int read_eigensystem_timeslice ( double **v, int numV, char*filename) {

  const size_t bytes = _GVI(1) * sizeof( double );
  const unsigned int VOL3 = LX*LY*LZ;
  int exitstatus;
  double ratime, retime;
 
  ratime = _GET_TIME;

  FILE *ifs = fopen ( filename, "r" );
  if ( ifs == NULL ) {
    fprintf( stderr, "[read_eigensystem_timeslice] Error from fopen %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  for ( int l = 0; l < numV; l++ ) {
    long int loffset = l * _GVI(LX_global*LY_global*LZ_global) * sizeof(double);

    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {

      unsigned int idx =  ( x1 * LY + x2 ) * LZ + x3;

      long int xoffset = _GVI( ( (long int)(x1 + g_proc_coords[1] * LX)*LY_global + (x2 + g_proc_coords[2] * LY) ) * LZ_global + (x3 + g_proc_coords[3] * LZ ) ) * sizeof(double);

      if ( ( exitstatus = fseek ( ifs, loffset + xoffset, SEEK_SET ) ) != 0 ) {
        fprintf ( stderr, "[read_eigensystem_timeslice] Error from fseek, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        return(2);
      }

      if ( fread ( v[l]+_GVI(idx), bytes, 1, ifs ) != 1 ) {
        fprintf ( stderr, "[read_eigensystem_timeslice] Error from fread %s %d\n", __FILE__, __LINE__ );
        return(2);
      }
    }}}
  }

  fclose ( ifs );

#if 0
  // TEST
  for ( int l = 0; l < numV; l++ ) {

    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      fprintf( stdout, "# [read_eigensystem_timeslice] l %3d x %3d %3d %3d\n", l, x1+g_proc_coords[1] * LX, x2+g_proc_coords[2] * LY, x3+g_proc_coords[3] * LZ );
      unsigned int idx = ( x1 * LY + x2 ) * LZ + x3;
      for ( int m = 0; m < 3; m++ ) {
        fprintf ( stdout, "  %3d %25.16e %25.16e\n", m, v[l][_GVI(idx)+2*m], v[l][_GVI(idx)+2*m+1] );
      }
    }}}
  }
  // END OF TEST
#endif  // of if 0

#ifdef HAVE_MPI
  if ( MPI_Barrier ( g_cart_grid ) != MPI_SUCCESS ) {
    fprintf ( stderr, "[read_eigensystem_timeslice] Error from MPI_Barrier\n" );
    return(3);
  }
#endif

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [] time for read_eigensystem_timeslice = %e seconds\n", retime-ratime );

  return(0);
}  /* end of read_eigensystem_timeslice */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int write_eigensystem_timeslice ( double **v, int numV, char*filename) {

#ifndef HAVE_MPI
  const size_t bytes = _GVI(1) * sizeof( double );
  const unsigned int VOL3 = LX*LY*LZ;
  const size_t items = VOL3 * numV;
  int exitstatus;

  FILE *ifs = fopen ( filename, "w" );
  if ( ifs == NULL ) {
    fprintf( stderr, "[write_eigensystem_timeslice] Error from fopen %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( fwrite ( v[0], bytes, items, ifs ) != items ) {
    fprintf ( stderr, "[write_eigensystem_timeslice] Error from fwrite %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

  fclose ( ifs );

  /* TEST */
  for ( int l = 0; l < numV; l++ ) {

    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      fprintf( stdout, "# [write_eigensystem_timeslice] l %3d x %3d %3d %3d\n", l, x1+g_proc_coords[1] * LX, x2+g_proc_coords[2] * LY, x3+g_proc_coords[3] * LZ );
      unsigned int idx = ( x1 * LY + x2 ) * LZ + x3;
      for ( int m = 0; m < 3; m++ ) {
        fprintf ( stdout, "  %3d %25.16e %25.16e\n", m, v[l][_GVI(idx)+2*m], v[l][_GVI(idx)+2*m+1] );
      }
    }}}
  }
  /* END OF TEST */


  return(0);
#else
  return(1);
#endif
}  /* end of write_eigensystem_timeslice */


/***********************************************************/
/***********************************************************/
}  /* end of namespace cvc */
