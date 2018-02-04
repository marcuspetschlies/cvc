/***************************************************
 * distillation_vertices.cpp
 ***************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "io_utils.h"
#include "read_input_parser.h"
#include "cvc_utils.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "iblas.h"
#include "project.h"
#include "matrix_init.h"
#include "distillation_vertices.h"


namespace cvc {


/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * calculate V^+ D^d_k V, where
 *   k = x, y, z
 *   d = fwd, bwd
 *
 *
 ***********************************************************************************************/
int distillation_vertex_displacement ( double**V, int numV, int momentum_number, int (*momentum_list)[3], char*prefix, char*tag, int io_proc, double *gauge_field, int timeslice ) {
  
  const unsigned int VOL3 = LX*LY*LZ;
  const size_t sizeof_colorvector_field_timeslice = _GVI(VOL3) * sizeof(double);

  int exitstatus;
  char filename[200];

  double ratime, retime, aff_retime, aff_ratime, total_ratime, total_retime;

  /***********************************************
   *variables for blas interface
   ***********************************************/
  double _Complex Z_1 = 1.;
  double _Complex Z_0 = 0.;

  char CHAR_N = 'N', CHAR_C = 'C';
  int INT_M = numV, INT_N = numV, INT_K = _GVI(VOL3)/2;

#ifdef HAVE_LHPC_AFF
  AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_key[200];
  char *aff_status_str;

  total_ratime = _GET_TIME;

  /***********************************************
   * writer for aff output file
   ***********************************************/
  if(io_proc >= 1) {
    sprintf(filename, "%s.%.4d.t%.2d.aff", prefix, Nconf, g_proc_coords[0]  );
    if ( io_proc == 2 ) fprintf(stdout, "# [distillation_vertex_displacement] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[distillation_vertex_displacement] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return(15);
    }
  
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[distillation_vertex_displacement] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }
#else
  FILE *ofs = NULL;
  if ( io_proc >= 1 ) {
    sprintf( filename, "%s.t%.2d.dat", prefix, g_proc_coords[0] );
    ofs = fopen ( filename, "w" );
    if( ofs == NULL ) {
      fprintf(stderr, "[distillation_vertex_displacement] Error from open for filename %s %s %d\n", filename, __FILE__, __LINE__);
      return(1);
    }
  }
  size_t write_count = numV * numV;

#endif

  /***********************************************/
  /***********************************************/

  /***********************************************
   * calculate W
   ***********************************************/
  double **W = NULL, **work = NULL;
  double _Complex **vv = NULL;

  exitstatus = init_2level_buffer ( &W, numV, _GVI(VOL3) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[distillation_vertex_displacement] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &work, 2, _GVI(VOL3+RAND3) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[distillation_vertex_displacement] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_zbuffer ( &vv, numV, numV );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[distillation_vertex_displacement] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }


  /***********************************************/
  /***********************************************/

  /***********************************************
   * loop on fbwd
   ***********************************************/
  for ( int fbwd = 0; fbwd < 2; fbwd++ ) {

    /***********************************************
     * loop on directions
     ***********************************************/
    for ( int mu = 1; mu < 4; mu++ ) {

      ratime = _GET_TIME;
      for ( int i = 0; i < numV; i++ ) {
        memcpy ( work[0], V[i], sizeof_colorvector_field_timeslice );
        apply_displacement_colorvector ( W[i], work[0], mu, fbwd, gauge_field, timeslice );
      }
      retime = _GET_TIME;
      if ( io_proc == 2 ) fprintf ( stdout, "# [distillation_vertex_displacement] time for displacement = %e seconds\n", retime-ratime );

      /***********************************************
       * phases for momentum projection
       ***********************************************/
      double *phase = NULL;
      exitstatus = init_1level_buffer ( &phase, VOL3 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[distillation_vertex_displacement] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      /***********************************************
       * loop on momenta
       ***********************************************/
      for ( int imom = 0; imom < momentum_number; imom++ ) {

        make_lexic_phase_field_3d ( phase, momentum_list[imom] );

        /***********************************************
         * multiply with momentum phase 
         ***********************************************/
        for ( int k = 0; k < numV; k++ ) {
          colorvector_field_eq_colorvector_field_ti_complex_field ( W[k], W[k], phase, VOL3 );
        }

        /***********************************************
         * V^+ D(p) V
         ***********************************************/
        _F(zgemm) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(V[0]), &INT_K, (double _Complex*)(W[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);

        /***********************************************
         * add half-link phase shift
         ***********************************************/
        int LL[3] = {LX, LY, LZ};
        double dtmp = ( 1 - 2*fbwd )* momentum_list[imom][mu-1] * M_PI / (double)LL[mu-1];
        double ztmp[2] = { cos(dtmp), sin(dtmp) };
        complex_field_ti_eq_co ( (double*)(vv[0]), ztmp, numV*numV);


#ifdef HAVE_MPI
        ratime = _GET_TIME;
        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        double *vvx = NULL;
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[distillation_vertex_displacement] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[distillation_vertex_displacement] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [distillation_vertex_displacement] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to file
         ***********************************************/
        if ( io_proc >= 1 ) {
          aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/v-Dp-v/t%.2d/fbwd%d/mu%d/px%.2dpy%.2dpz%.2d", tag, timeslice+g_proc_coords[0]*T, fbwd, mu, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2] );
          
          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[distillation_vertex_displacement] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[distillation_vertex_displacement] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif
          aff_retime = _GET_TIME;
          fprintf(stdout, "# [distillation_vertex_displacement] time for writing = %e\n", aff_retime-aff_ratime);

        }

      }  /* end of loop on momenta */

      fini_1level_buffer ( &phase );

    }  /* end of loop on directions */
  
  }  /* end of loop on fbwd */

  /***********************************************/
  /***********************************************/


#ifdef HAVE_LHPC_AFF
  /***********************************************************
  * close the AFF writer
  ***********************************************************/
  if( io_proc >= 1 ) {
    aff_status_str = (char*)aff_writer_close ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[distillation_vertex_displacement] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return(32);
    }
  }  /* end of if io_proc >= 1 */
#else
  /***********************************************
   * close output file
   ***********************************************/
  if ( io_proc >= 1 ) fclose ( ofs );
#endif

  /***********************************************/
  /***********************************************/

  /***********************************************
   * deallocate auxilliary fields
   ***********************************************/
  fini_2level_buffer ( &W );
  fini_2level_buffer ( &work );
  fini_2level_zbuffer ( &vv );

  /***********************************************/
  /***********************************************/

#ifdef HAVE_MPI
  /***********************************************
   * mpi barrier and total time
   ***********************************************/
  if ( ( exitstatus =  MPI_Barrier ( g_cart_grid ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[distillation_vertex_displacement] Error from MPI_Barrier, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(6);
  }
#endif
  total_retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf(stdout, "\n# [distillation_vertex_displacement] time for gsp_calculate_v_dag_gamma_p_w_block = %e seconds\n", total_retime-total_ratime);
  fflush ( stdout );

  return(0);

}  /* end of distillation_vertex_displacement */

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */
