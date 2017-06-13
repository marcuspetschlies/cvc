/****************************************************
 * piN2piN_diagrams.c
 * 
 * Mon Jun  5 15:59:40 CDT 2017
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
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "contract_diagrams.h"
#include "gamma.h"


using namespace cvc;


/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  fprintf(stdout, "Code to perform contractions for piN 2-pt. function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  const int n_c=3;
  const int n_s=4;
  const int max_num_diagram = 6;


  int c, i, k, i_src, isample;
  int filename_set = 0;
  int exitstatus;
  int it, ir, is;
  int op_id_up= -1, op_id_dn = -1;
  int gsx[4], sx[4];
  int source_proc_id = 0;
  char filename[200];
  double ratime, retime;
  double plaq_m = 0., plaq_r = 0.;
  double *spinor_work[2];
  unsigned int VOL3;
  size_t sizeof_spinor_field = 0, sizeof_spinor_field_timeslice = 0;
  spinor_propagator_type **conn_X=NULL;
  double ****buffer=NULL;
  int io_proc = -1;
  double **propagator_list_up = NULL, **propagator_list_dn = NULL, **sequential_propagator_list = NULL, **stochastic_propagator_list = NULL,
         **stochastic_source_list = NULL;
  double *gauge_field_smeared = NULL, *tmLQCD_gauge_field = NULL;

/*******************************************************************
 * Gamma components for the piN and Delta:
 *                                                                 */
  /* vertex i2, gamma_5 only */
  const int gamma_i2_number = 1;
  int gamma_i2_list[1]      = {  5 };
  double gamma_i2_sign[1]   = { +1 };

  /* vertex f2, gamma_5 and id,  vector indices and pseudo-vector */
  const int gamma_f2_number = 2;
  int gamma_f2_list[2]      = {  5,  4 };
  double gamma_f2_sign[2]   = { +1, +1 };

  /* vertex c, vector indices and pseudo-vector */
  const int gamma_c_number = 6;
  int gamma_c_list[6]       = {  1,  2,  3,  7,  8,  9 };
  double gamma_c_sign[6]    = { +1, +1, +1, +1, +1, +1 };


  /* vertex f1 for nucleon-type, C g5, C, C g5 g0, C g0 */
  const int gamma_f1_nucleon_number = 4;
  int gamma_f1_nucleon_list[4]      = { 14, 11,  8,  2 };
  double gamma_f1_nucleon_sign[4]   = { +1, +1, +1, -1 };

  /* vertex f1 for Delta-type operators, C gi, C gi g0 */
  const int gamma_f1_delta_number = 6;
  int gamma_f1_delta_list[6]      = { 9,  0,  7, 13,  4, 15 };
  double gamma_f1_delta_sign[6]   = {+1, +1, -1, -1, +1, +1 };


  int num_component_max = 9;
/*
 *******************************************************************/

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char * aff_status_str;
  char aff_tag[200];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  if(g_fermion_type == -1 ) {
    fprintf(stderr, "# [piN2piN_factorized] fermion_type must be set\n");
    exit(1);
  } else {
    fprintf(stdout, "# [piN2piN_factorized] using fermion type %d\n", g_fermion_type);
  }


#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_factorized] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

  VOL3 = LX*LY*LZ;
  sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  sizeof_spinor_field_timeslice = _GSI(VOL3)   * sizeof(double);


  /* loop on source locations */
  for(i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
 
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.sample%.2d.tsrc%.2d.aff", "piN_piN_oet", Nconf, isample, t_base );
      fprintf(stdout, "# [piN2piN_factorized] writing data to file %s\n", filename);
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_factorized] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      }
    }  /* end of if io_proc == 2 */

    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global; 
 
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

      ratime = _GET_TIME;
      get_point_source_info (gsx, sx, &source_proc_id);

      exitstatus= init_2level_zbuffer ( &v3p, T_global, 12 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_2level_zbuffer ( &v2p, T_global,  192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_zbuffer ( &dgr, T_global, 4, 4 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {

        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {


        }  /* end of loop on samples */
      }  /* end of loop on gf1  */

      fini_2level_buffer ( &v1 );
      fini_2level_buffer ( &v2 );
      fini_3level_buffer ( &vp );
      free_fp_field ( &fp  );
      free_fp_field ( &fp2 );
    } /* end of loop on coherent source timeslices */

    /***********************************************************
     * sequential propagator
     ***********************************************************/

    /* loop on sequential source momenta */
    for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

      /***********************************************************
       * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
       ***********************************************************/
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] sequential inversion fpr pi2 = (%d, %d, %d)\n", 
      g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

      double **prop_list = (double**)malloc(g_coherent_source_number * sizeof(double*));
      if(prop_list == NULL) {
        fprintf(stderr, "[piN2piN_factorized] Error from malloc\n");
        EXIT(43);
      }

      ratime = _GET_TIME;
      for(is=0;is<n_s*n_c;is++) {

        /* extract spin-color source-component is from coherent source dn propagators */
        for(i=0; i<g_coherent_source_number; i++) {
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] using dn prop id %d / %d\n", (i_src * g_coherent_source_number + i), (i_src * g_coherent_source_number + i)*n_s*n_c + is);
          prop_list[i] = propagator_list_dn[i * n_s*n_c + is];
        }

        /* build sequential source */
        exitstatus = init_coherent_sequential_source(spinor_work[0], prop_list, gsx[0], g_coherent_source_number, g_seq_source_momentum_list[iseq_mom], 5);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_coherent_sequential_source, status was %d\n", exitstatus);
          EXIT(14);
        }

        /* source-smear the coherent source */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

        /* tm-rotate sequential source */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
        }

        memset(spinor_work[1], 0, sizeof_spinor_field);
        /* invert */
        exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        /* tm-rotate at sink */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
        }

        /* sink-smear the coherent-source propagator */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

        memcpy( sequential_propagator_list[is], spinor_work[1], sizeof_spinor_field);

      }  /* end of loop on spin-color component */
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] time for seq propagator = %e seconds\n", retime-ratime);

      free(prop_list);
 
      /***********************************************
       * contractions involving sequential propagator
       ***********************************************/
      double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
      fermion_propagator_type *fp = NULL, *fp2 = NULL;

      exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      fp = create_fp_field ( VOLUME );
      assign_fermion_propagator_from_spinor_field ( fp, sequential_propagator_list, VOLUME);

      for ( int i = 0; i < gamma_f2_number; i++ ) {
        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {

          /*****************************************************************
           * xi - gf2 - ud
           *****************************************************************/
          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%2d-ud/sample%2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, stochastic_source_list[i_sample], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#if 0
          /*****************************************************************
           * phi - gf2 - ud
           *****************************************************************/
          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%2d-ud/sample%2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, stochastic_propagator_list[i_sample], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif

        }  /* end of loop on samples */
      }  /* end of loop on gf2  */

      fini_2level_buffer ( &v3 );
      fini_3level_buffer ( &vp );

#if 0
      exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      fp2 = create_fp_field ( VOLUME );

      for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        get_point_source_info (gsx, sx, &source_proc_id);

        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);

        for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {
          for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {
  
            /*****************************************************************
             * xi - gf1 - ud 
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_source_list[i_sample], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * xi - gf1 - u - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%2d-ud-u/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3], 
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - ud
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_propagator_list[i_sample], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * phi - gf1 - ud - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%2d-ud-u/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************/
            /*****************************************************************/

            /*****************************************************************
             * xi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_source_list[i_sample], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * xi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%2d-u-ud/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_propagator_list[i_sample], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * phi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%2d-u-ud/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

          }  /* end of loop on samples */
        }  /* end of loop on gamma_f1 */

        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_dn[i_coherent * n_s*n_c]), VOLUME);

        for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {
          for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {
  
            /*****************************************************************
             * xi - gf1 - ud
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_source_list[i_sample], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }

            /*****************************************************************
             * xi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%2d-ud-d/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - ud
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_propagator_list[i_sample], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * phi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%2d-ud-d/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * xi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_source_list[i_sample], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * xi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%2d-d-ud/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, stochastic_propagator_list[i_sample], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * phi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%2d-d-ud/sample%2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f2_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
          }  /* end of loop on samples */
        }  /* end of loop on gf1  */
      }  /* end of loop on coherent source timeslices */

      fini_2level_buffer ( &v1 );
      fini_2level_buffer ( &v2 );
      fini_3level_buffer ( &vp );
      free_fp_field ( &fp  );
      free_fp_field ( &fp2 );

#endif
    }  /* end of loop on sequential momentum list */

  }  /* end of loop on base source locations */

  fini_2level_buffer ( &sequential_propagator_list );
  fini_2level_buffer ( &stochastic_propagator_list );
  fini_2level_buffer ( &stochastic_source_list );
  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );

#if 0

  /***********************************************
   ***********************************************
   **
   ** stochastic contractions using the 
   **   one-end-trick
   **
   ***********************************************
   ***********************************************/
  exitstatus = init_2level_buffer ( &stochastic_propagator_list, 1, _GSI(VOLUME) );
  if ( existatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &stochastic_source_list, 4, _GSI(VOLUME) );
  if ( existatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_up, 12, _GSI(VOLUME) );
  if ( existatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_dn, 12, _GSI(VOLUME) );
  if ( existatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }


  /* loop on oet samples */
  for(isample=0; isample < g_nsample_oet; isample++) {

    for(i_src=0; i_src < g_source_location_number; i_src++) {
      int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * open aff output file
       ***********************************************/
      if(io_proc == 2) {
        sprintf(filename, "%s.%.4d.sample%.2d.tsrc%.2d.aff", "piN_piN_oet", Nconf, isample, t_base );
        fprintf(stdout, "# [piN2piN_factorized] writing data to file %s\n", filename);
        affw = aff_writer(filename);
        aff_status_str = (char*)aff_writer_errstr(affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN_factorized] Error from aff_writer, status was %s\n", aff_status_str);
          EXIT(4);
        }
      }  /* end of if io_proc == 2 */
#endif

      for(i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {

        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        exitstatus = point_source_propagator ( propagator_list_up, gsx, op_id_up, 1, 1, gauge_field_smeared );
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d\n", exitstatus);
          EXIT(12);
        }

        exitstatus = point_source_propagator ( propagator_list_dn, gsx, op_id_dn, 1, 1, gauge_field_smeared );
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d\n", exitstatus);
          EXIT(12);
        }


        if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, 1)) != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_timeslice_source_oet, status was %d\n", exitstatus);
          EXIT(63);
        }

        for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], g_seq_source_momentum_list[iseq_mom], 0) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_timeslice_source_oet, status was %d\n", exitstatus);
            EXIT(64);
          }

          /* nonzero-momentum propagator */
          for( int i = 0; i < 4; i++) {
            
            double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
            fermion_propagator_type *fp = NULL, *fp2 = NULL;

            STOPPED HERE
            exitstatus = init_2level_buffer ( &v3, 


            memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

            /* source-smearing stochastic momentum source */
            exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

            /* tm-rotate stochastic source */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
            }

            memset(spinor_work[1], 0, sizeof_spinor_field);
            exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
            if(exitstatus != 0) {
              fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
              EXIT(44);
            }

            /* tm-rotate stochastic propagator at sink */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
            }

            /* sink smearing stochastic propagator */
            exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

            memcpy( stochastic_propagator_list[4+i], spinor_work[1], sizeof_spinor_field);
          }

        }  /* end of loop on sequential source momenta pi2 */
      }  /* end of loop on coherent sources */

#ifdef HAVE_LHPC_AFF
      if(io_proc == 2) {
        aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN_factorized] Error from aff_writer_close, status was %s\n", aff_status_str);
          EXIT(111);
        }
        if(aff_buffer != NULL) free(aff_buffer);
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on base sources */

  }  /* end of loop on oet samples */

  fini_2level_buffer ( &stochastic_propagator_list );
  fini_2level_buffer ( &stochastic_source_list );
  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );
#endif

  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if(g_gauge_field != NULL) free(g_gauge_field);
#endif

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();

  if( gauge_field_smeared != NULL ) free(gauge_field_smeared);
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_factorized] %s# [piN2piN_factorized] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_factorized] %s# [piN2piN_factorized] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
