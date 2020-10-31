/***************************************************
 * piN2piN_projection
 * 
 * PURPOSE:
 *   originally copied from piN2piN_correlators.cpp
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
#include <sys/time.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>

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

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM
#include <hdf5.h>


#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "cvc_timer.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "table_init_z.h"
#include "table_init_2pt.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"
#include "little_group_projector_set.h"

#define MAX_UDLI_NUM 1000
#define EPS 1e-14
#define ANNIHILATION 1 
#define CREATION 0
using namespace cvc;
/* Taking care of orthogonalizing Projector matrix */

double _Complex scalar_product( double _Complex *vec1, double _Complex *vec2, int N){
   double _Complex ret=0.;
   for (int j=0; j<N; ++j){
     ret+=conj(vec1[j])*vec2[j];
   }
   return ret;
}

/* Auxiliary routine, that performs a rank revealed QR algorithm */


/***********************************************************
 * main program
 * This code is supposed to be produce table of projected correlation
 * functions for piN-D system
 ***********************************************************/
int main(int argc, char **argv) {
 
#define _ZCOEFF_EPS 8.e-12


  int c;
  int filename_set = 0;
  int exitstatus;
  int check_reference_rotation = 0;
  char filename[400];
  char tagname[400];
  double ratime, retime;
  FILE *ofs = NULL;


  struct timeval ta, tb;

  /***********************************************************
   * initialize MPI if used
   ***********************************************************/
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  /***********************************************************
   * evaluate command line arguments
   ***********************************************************/
  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_reference_rotation = 1;
      fprintf ( stdout, "# [piN2piN_projection] check_reference_rotation set to %d\n", check_reference_rotation );
      break;
    case 'h':
    case '?':
    default:
      exit(1);
      break;
    }
  }

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_projection] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************
   * package-own initialization of MPI parameters
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_projection] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_projection] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();

  /***********************************************************
   * TEST: report size of twopoint_function_type
   ***********************************************************/
  if ( io_proc == 0 ) {
    fprintf ( stdout, "# [piN2piN_projection] sizeof twopoint_function_type = %lu\n", sizeof ( twopoint_function_type ) );
  }



  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ***********************************************************/
  init_gamma_matrix ("plegma");

  /******************************************************
   * set gamma matrices
   *   tmLQCD counting
   ******************************************************/
  gamma_matrix_type gamma[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(gamma[i]), i, 1. );
  }


   
  /******************************************************
   * loop on 2-point functions
   ******************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    printf("# [piN2piN_projection_apply] start analyzing twopoint function index %d\n", i2pt);

    for (int ii=0; ii< g_twopoint_function_list[i2pt].nlistmomentumf1; ++ii){

      printf("# [piN2piN_projection_apply] pf1 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf1list[ii][0] , g_twopoint_function_list[i2pt].pf1list[ii][1] , g_twopoint_function_list[i2pt].pf1list[ii][2] );
      printf("# [piN2piN_projection_apply] pf2 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf2list[ii][0] , g_twopoint_function_list[i2pt].pf2list[ii][1] , g_twopoint_function_list[i2pt].pf2list[ii][2] );

    }


    /****************************************************
     * read little group parameters, this is only needed in order to 
     * know the number of rows of the irrep, the number of mus
     ****************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
    little_group_type little_group;
    printf("# [piN2piN_projection] little group %s\n",g_twopoint_function_list[i2pt].group);
    if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "piN2piN_projection", "little_group_read", g_cart_id == 0 );

    sprintf ( filename, "little_group_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    little_group_show ( &little_group, ofs, 1 );
    fclose ( ofs );

    /****************************************************
     * initialize and set projector 
     * for current little group and irrep
     ****************************************************/
    little_group_projector_type projector;
    if ( ( exitstatus = init_little_group_projector ( &projector ) ) != 0 ) {
      fprintf ( stderr, "# [piN2piN_projection] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
    

    /****************************************************
     * parameters for setting the projector
     ****************************************************/
    int ref_row_target          = -1;     // no reference row for target irrep
    int * ref_row_spin          = NULL;   // no reference row for spin matrices
    int refframerot             = -1;     // reference frame rotation
                                          //   added below
    int row_target              = -1;     // no target row
    int cartesian_list[1]       = { 0 };  // not cartesian
    int parity_list[1]          = { 1 };  // intrinsic parity is +1
    const int ** momentum_list  = NULL;   // no momentum list given
    int bispinor_list[1]        = { 1 };  // bispinor yes
    int J2_list[1]              = { 1 };  // spin 1/2
    int Pref[3] = {-1,-1,-1};

    int const Ptot[3] = {
      g_twopoint_function_list[i2pt].pf1list[0][0] + g_twopoint_function_list[i2pt].pf2list[0][0] , 
      g_twopoint_function_list[i2pt].pf1list[0][1] + g_twopoint_function_list[i2pt].pf2list[0][1] , 
      g_twopoint_function_list[i2pt].pf1list[0][2] + g_twopoint_function_list[i2pt].pf2list[0][2]  };


    /****************************************************
     * do we need a reference frame rotation ?
     ****************************************************/
    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt, 
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }


    fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt,
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);



    fflush ( stdout );

    if (g_twopoint_function_list[i2pt].nlistmomentumf1 > 1 && g_twopoint_function_list[i2pt].nlistmomentumf1 != g_twopoint_function_list[i2pt].nlistmomentumf2){
      fprintf( stderr, "[piN2piN_projection_tables] Error different f1 and f2 momenta\n");
      exit(1);
    }
    int check_consistency=0;
    for (int ii=0; ii<g_twopoint_function_list[i2pt].nlistmomentumf1; ++ii){
      if ((Ptot[0]!=( g_twopoint_function_list[i2pt].pf1list[ii][0] + g_twopoint_function_list[i2pt].pf2list[ii][0])) || (Ptot[1]!=( g_twopoint_function_list[i2pt].pf1list[ii][1] + g_twopoint_function_list[i2pt].pf2list[ii][1])) || (Ptot[2]!=( g_twopoint_function_list[i2pt].pf1list[ii][2] + g_twopoint_function_list[i2pt].pf2list[ii][2])) ){
        check_consistency=1;
        break;
      }  
    }
    if (check_consistency == 1){
      fprintf(stderr,"[piNpiN_projection_table] check the input values of momenta, some of them are not giving the correct total momentum\n");
      exit(1);
    }

    /****************************************************
     * set the projector with the info we have
     ****************************************************/
    exitstatus = little_group_projector_set (
        &projector,
        &little_group,
        g_twopoint_function_list[i2pt].irrep ,
        row_target,
        1,
        J2_list,
        momentum_list,
        bispinor_list,
        parity_list,
        cartesian_list,
        ref_row_target,
        ref_row_spin,
        g_twopoint_function_list[i2pt].type,
        refframerot );

    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[piN2piN_projection_tables] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( filename, "little_group_projector_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    exitstatus = little_group_projector_show ( &projector, ofs, 1 );
    fclose ( ofs );

    
   
    /****************************************************
     * check, that projector has correct d-vector
     ****************************************************/
    if ( ( projector.P[0] != Ptot[0] ) || ( projector.P[1] != Ptot[1] ) || ( projector.P[2] != Ptot[2] ) ) {
      fprintf ( stderr, "[piN2piN_projection] Error, projector P != Ptot\n" );
      EXIT(12);
    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_projection] projector P == Ptot\n" );
    }


    int const irrep_dim = projector.rtarget->dim;

    /******************************************************
     * Open the table for creation and annihilation interpolating operator coefficients 
     *****************************************************/

    for ( int ibeta = 0; ibeta < irrep_dim; ibeta++ ) {

      for ( int imu = 0; imu < irrep_dim; imu++ ) {


       /***********************************************************
        * read data block from h5 file
        ***********************************************************/

        int pfnx=g_twopoint_function_list[i2pt].pf1list[0][0];
        int pfny=g_twopoint_function_list[i2pt].pf1list[0][1];
        int pfnz=g_twopoint_function_list[i2pt].pf1list[0][2];

        int pfpx=g_twopoint_function_list[i2pt].pf2list[0][0];
        int pfpy=g_twopoint_function_list[i2pt].pf2list[0][1];
        int pfpz=g_twopoint_function_list[i2pt].pf2list[0][2];

        snprintf ( filename, 400, "projection_coefficients_piN_group_%s_irrep_%s.h5",
                          g_twopoint_function_list[i2pt].group,
                          g_twopoint_function_list[i2pt].irrep );
        snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/Nreplicas_Nps",  Ptot[0],Ptot[1], Ptot[2], pfpx*pfpx+pfpy*pfpy+pfpz*pfpz, pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta );

        int *Nps_Nreplica_piN= init_1level_itable( 2 );


        exitstatus = read_from_h5_file ( (void*)(Nps_Nreplica), filename, tagname, io_proc, 1 );

        int ** momtable = init_2level_itable ( Nps_Nreplica[1], 6 );

        snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1], Ptot[2], pfpx*pfpx+pfpy*pfpy+pfpz*pfpz, pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta );

        exitstatus = read_from_h5_file ( (void*)(momtable[0]), filename, tagname, io_proc, 1 );

        double ***projection_coeff_a_ORT= init_3level_dtable( Nps_Nreplica_piN[0], Nps_Nreplica_piN[1]*spin1212dimension*spin1dimension, 2);

        double ***projection_coeff_c_ORT= init_3level_dtable( Nps_Nreplica_piN[1]*spin1212dimension*spin1dimension, Nps_Nreplica_piN[0], 2);

        snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/a_data_ort",  Ptot[0],Ptot[1], Ptot[2], pfpx*pfpx+pfpy*pfpy+pfpz*pfpz, pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta );

        exitstatus = read_from_h5_file ( (void*)(projection_coeff_a_ORT[0][0]), filename, tagname, io_proc, 1 );

        snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/c_data_ort",  Ptot[0],Ptot[1], Ptot[2], pfpx*pfpx+pfpy*pfpy+pfpz*pfpz, pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta );

        exitstatus = read_from_h5_file ( (void*)(projection_coeff_c_ORT[0][0]), filename, tagname, io_proc, 1 );

        int nternal_dimension =  Nps_Nreplica_piN[1]*spin1212dimension*spin1dimension;

        for (int nreplicum_source=0; nreplicum_source < Nps_Nreplica_piN[0] ; ++nreplicum_source ) {

          for ( int nreplicum_sink=0; nreplicum_sink < Nps_Nreplica_piN[0] ; ++nreplicum_sink ) {


            double **projected_correlation_function=init_2level_dtable ( g_twopoint_function_list[i2pt].T, 2);

            for ( int i1_source =0; i1_source < Nps_Nreplica_piN[1] ; ++i1_source ) {
    
              for ( int i2_source=0; i2_source < spin1dimension ; ++i2_source ) {

                for ( int i1_sink =0; i1_sink < Nps_Nreplica_piN[1] ; ++i1_sink ) {

                  for ( int i2_sink =0; i2_sink < spin1dimension ; ++i2_sink ) {

                    int pf1x=momtable[i1_sink][0];
                    int pf1y=momtable[i1_sink][1];
                    int pf1z=momtable[i1_sink][2];
                    int pf2x=momtable[i1_sink][3];
                    int pf2y=momtable[i1_sink][4];
                    int pf2z=momtable[i1_sink][5];

                    int pi1x=momtable[i1_source][0];
                    int pi1y=momtable[i1_source][1];
                    int pi1z=momtable[i1_source][2];
                    int pi2x=momtable[i1_source][3];
                    int pi2y=momtable[i1_source][4];
                    int pi2z=momtable[i1_source][5];

                    int gamma_f1=g_twopoint_function_list[i2pt].gammalistf1[i2_sink];
                    int gamma_i1=g_twopoint_function_list[i2pt].gammalistf1[i2_source];
 
                    snprintf ( tagname, 400, "/sx%.02dsy%.02dsz%.02dst%03d/gf25/pf2x%.02dpf2y%.02dpf2z%.02d/gf1%s,%s/pf1x%.02dpf1y%.02dpf1z%.02d/gi25/pi2x%.02dpi2y%.02dpi2z%.02d/gi1%s,%s/pi1x%.02dpi1y%.02dpi1z%.02d/piN", source_coords_list[k][1],
                                                        source_coords_list[k][2],
                                                        source_coords_list[k][3],
                                                        source_coords_list[k][0],
                                                        pf2x,
                                                        pf2y,
                                                        pf2z,
                                                        gamma_string_list_sink[sink_gamma],
                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                        pf1x,
                                                        pf1y,
                                                        pf1z,
                                                        pi2x,
                                                        pi2y,
                                                        pi2z,
                                                        gamma_string_list_source[source_gamma],
                                                        ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                        pi1x,
                                                        pi1y,
                                                        pi1z);

                    snprintf ( filename, 400, "%s%04d_PX%.02dPY%.02dPZ%.02d_piN.h5",
                                                        filename_prefix,
                                                        Nconf,
                                                        momentum_orbit_pref[i_total_momentum][0],
                                                        momentum_orbit_pref[i_total_momentum][1],
                                                        momentum_orbit_pref[i_total_momentum][2]);
                    double ***correlation_function=init_3level_dtable( g_twopoint_function_list[i2pt].T,g_twopoint_function_list[i2pt].d ,2 );

                    exitstatus = read_from_h5_file ( (void*)(momtable[0]), filename, tagname, io_proc, 1 );

                    for ( int i3_source =0; i3_source < spin1212dimension; ++i3_source ) {


                      for ( int i3_sink =0; i3_sink < spin1212dimension ; ++i3_sink ) {

                          for (int t=0; t<t; ++t){

                            projected_correlation_function[t][0]+=(projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension*spin1212dimenison+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension*spin1212dimenison+i2_source*spin1212dimension+i3_source][nreplicum_source][0]+projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension*spin1212dimenison+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension*spin1212dimenison+i2_source*spin1212dimension+i3_source][nreplicum_source][1])*correlation_function[t][i3_source*4+i3_sink][0];
                            projected_correlation_function[t][1]+=(projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension*spin1212dimenison+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension*spin1212dimenison+i2_source*spin1212dimension+i3_source][nreplicum_source][0]+projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension*spin1212dimenison+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension*spin1212dimenison+i2_source*spin1212dimension+i3_source][nreplicum_source][1])*correlation_function[t][i3_source*4+i3_sink][0]; 
                          }/*time*/

                      }/*i3 sink */
                    }/*i3_source */
                  }/*i2_sink */
                }/*i1_sink */
              }/*i2_source*/
            }/*i1_source*/

          } /*nreplicum sink */

        } /*nreplicum source */

      } /*beta */

    } /*mu*/ 

    /******************************************************
     * deallocate space inside little_group
     ******************************************************/
    little_group_fini ( &little_group );

    /******************************************************
     * deallocate space inside projector
     ******************************************************/
    fini_little_group_projector ( &projector );

  }  // end of loop on 2-point functions

  /******************************************************/
  /******************************************************/
#if 0
#endif  /* of if 0 */

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
