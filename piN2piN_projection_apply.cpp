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
#include "contractions_io.h"
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

char *convert_gamma_to_string ( int gamma ) {

   char *ret_string=(char *)malloc(sizeof(char)*100);
   if ( gamma == 9){
     snprintf(ret_string,100,"cg1,1");
   }
   else if (gamma== 0){
     snprintf(ret_string,100,"cg2,1");
   }
   else if (gamma==7){
     snprintf(ret_string,100,"cg3,1");
   }
   else if (gamma==13){
     snprintf(ret_string,100,"cg1g4,1");
   }
   else if (gamma==4){
     snprintf(ret_string,100,"cg2g4,1");
   }
   else if (gamma==15){
     snprintf(ret_string,100,"cg3g4,1");
   }
   else if ( gamma==12){
     snprintf(ret_string,100,"cg1g4g5,5");
   }
   else if ( gamma==5) {
     snprintf(ret_string,100,"cg2g4g5,5");
   }
   else if (gamma==10) {
     snprintf(ret_string,100,"cg3g4g5,5");
   }
   else if (gamma==14) {
     snprintf(ret_string,100, "Cg5,1");
   }
   else if (gamma==11) {
     snprintf(ret_string,100, "C,5");
   }
   else if (gamma==8) {
     snprintf(ret_string,100, "Cg5g4,1");
   }
   else if (gamma==2) {
     snprintf(ret_string,100, "Cg4,5");
   }
   else {
     fprintf(stderr, "# [convert_gamma_to_string] Non recognized gamma in conversion\n");}
   return ret_string;

}

char *tagname_forgamma_multiplets ( int *gamma_multiplet, int length ){

   char *ret_string=(char *)malloc(sizeof(char)*100);
   if (length==3){
     if ((gamma_multiplet[0]==9) && (gamma_multiplet[1]==0)  && (gamma_multiplet[2]==7)){
       snprintf(ret_string, 100, "cgxyz");
     }
     else if ((gamma_multiplet[0]==13) && (gamma_multiplet[1]==4)  && (gamma_multiplet[2]==15)){
       snprintf(ret_string, 100, "cgxyzg0");
     }
     else if ((gamma_multiplet[0]==12) && (gamma_multiplet[1]==5)  && (gamma_multiplet[2]==10)){
       snprintf(ret_string, 100, "cgxyzg0g5");

     }
     else {
       fprintf(stderr, "# [tagname_forgamma_multiplets] No recognized multiplett %d %d %d\n", gamma_multiplet[0],gamma_multiplet[1],gamma_multiplet[2]);
       exit(1);
     }
   }
   else if (length==1) {
     if (gamma_multiplet[0]==14){
       snprintf(ret_string,100, "cg5");
     }
     else if  (gamma_multiplet[0]==11){
       snprintf(ret_string,100, "c");
     }
     else if  (gamma_multiplet[0]==8){
       snprintf(ret_string,100, "cg5g4");
     }
     else if  (gamma_multiplet[0]==2){
       snprintf(ret_string,100, "cg4");
     }
     else {
       fprintf(stderr, "# [tagname_forgamma_multiplets] No recognized multiplett \n");
       exit(1);
     }
  }
  else {
       fprintf(stderr, "# [tagname_forgamma_multiplets] No recognized length of mupliplett \n");
       exit(1);
  }
  return ret_string;

}

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

  int const momentum_orbit_pref[4][3] = { {0,0,0}, {0,0,1}, {1,1,0}, {1,1,1} };


  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] +  T_global ) %  T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************
   * total number of source locations, 
   * base x coherent
   *
   * fill a list of all source coordinates
   ******************************************************/
  int const source_location_number = g_source_location_number * g_coherent_source_number;
  int ** source_coords_list = init_2level_itable ( source_location_number, 4 );
  if( source_coords_list == NULL ) {
    fprintf ( stderr, "# [piN2piN_diagram_sum_per_type] Error from init_2level_itable %s %d\n", __FILE__, __LINE__ );
    EXIT( 43 );
  }
  for ( int ib = 0; ib < g_source_location_number; ib++ ) {
    g_source_coords_list[ib][0] = ( g_source_coords_list[ib][0] +  T_global ) %  T_global;
    g_source_coords_list[ib][1] = ( g_source_coords_list[ib][1] + LX_global ) % LX_global;
    g_source_coords_list[ib][2] = ( g_source_coords_list[ib][2] + LY_global ) % LY_global;
    g_source_coords_list[ib][3] = ( g_source_coords_list[ib][3] + LZ_global ) % LZ_global;

    int const t_base = g_source_coords_list[ib][0];

    for ( int ic = 0; ic < g_coherent_source_number; ic++ ) {
      int const ibc = ib * g_coherent_source_number + ic;

      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * ic ) % T_global;
      source_coords_list[ibc][0] = t_coherent;
      source_coords_list[ibc][1] = ( g_source_coords_list[ib][1] + (LX_global/2) * ic ) % LX_global;
      source_coords_list[ibc][2] = ( g_source_coords_list[ib][2] + (LY_global/2) * ic ) % LY_global;
      source_coords_list[ibc][3] = ( g_source_coords_list[ib][3] + (LZ_global/2) * ic ) % LZ_global;
    }
  }


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

  /* loop over the source positions */
  for ( int k = 0; k < source_location_number; k++ ) {

    for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

       printf("# [piN2piN_projection_apply] start analyzing twopoint function index %d\n", i2pt);

       for (int ii=0; ii< g_twopoint_function_list[i2pt].nlistmomentumf1; ++ii){

         printf("# [piN2piN_projection_apply] pf1 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf1list[ii][0] , g_twopoint_function_list[i2pt].pf1list[ii][1] , g_twopoint_function_list[i2pt].pf1list[ii][2] );
         printf("# [piN2piN_projection_apply] pf2 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf2list[ii][0] , g_twopoint_function_list[i2pt].pf2list[ii][1] , g_twopoint_function_list[i2pt].pf2list[ii][2] );

       }


       const int numberofnplets_source = g_twopoint_function_list[i2pt].number_of_gammas_i1/g_twopoint_function_list[i2pt].contniuum_spin_particle_source;

       const int spin1dimension_source = g_twopoint_function_list[i2pt].contniuum_spin_particle_source;

       const int numberofnplets_sink = g_twopoint_function_list[i2pt].number_of_gammas_f1/g_twopoint_function_list[i2pt].contniuum_spin_particle_sink;

       const int spin1dimension_sink   = g_twopoint_function_list[i2pt].contniuum_spin_particle_sink;

       const int spin1212dimension = g_twopoint_function_list[i2pt].d;

       hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
       herr_t      status;


       snprintf ( filename, 400, "Projected_corrfunc_sink_%s_source_%s_group_%s_irrep_%s.h5",
                         g_twopoint_function_list[i2pt].particlename_sink,
                         g_twopoint_function_list[i2pt].particlename_source,
                         g_twopoint_function_list[i2pt].group,
                         g_twopoint_function_list[i2pt].irrep);

       struct stat fileStat;
       if(stat( filename, &fileStat) < 0 ) {

         /* Open an existing file. */
         fprintf ( stdout, "# [test_hdf5] create new file %s\n",filename );
         file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
       } else {
         fprintf ( stdout, "# [test_hdf5] open existing file %s\n", filename );
         file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
       }


       /* Determining the dimension of the irrep */

       int const Ptot[3] = {
         g_twopoint_function_list[i2pt].pf1list[0][0] + g_twopoint_function_list[i2pt].pf2list[0][0] , 
         g_twopoint_function_list[i2pt].pf1list[0][1] + g_twopoint_function_list[i2pt].pf2list[0][1] , 
         g_twopoint_function_list[i2pt].pf1list[0][2] + g_twopoint_function_list[i2pt].pf2list[0][2]  };


       snprintf ( filename, 400, "projection_coefficients_%s_group_%s_irrep_%s.h5",
                            g_twopoint_function_list[i2pt].particlename_source,
                            g_twopoint_function_list[i2pt].group,
                            g_twopoint_function_list[i2pt].irrep );

       fprintf(stdout, "# [piN2piN_projection_apply]  determining dimension of the irrep\n");

       if ( strcmp(g_twopoint_function_list[i2pt].particlename_source, "piN") == 0) {
          snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_source[0], g_twopoint_function_list[i2pt].total_momentum_nucleon_source[0], 0, 0 );}
       else { 
          snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2],  0, 0 );       }

       int *tmp= init_1level_itable( 3 );

       exitstatus = read_from_h5_file ( (void*)(tmp), filename, tagname, io_proc, 1 );
       if (exitstatus != 0){
         fprintf(stderr, "Error in opening file for determining the dimension of irrep\n");
         exit(1);
       }

       const int irrep_dim=tmp[2];

       fini_1level_itable(&tmp);
       fprintf(stdout,"# [piN2piN_projection_apply] Irrep dimension=%d\n", irrep_dim);

       /******************************************************
        * Open the table for creation and annihilation interpolating operator coefficients 
        *****************************************************/
       snprintf(tagname,400,"/sx%0.2dsy%0.2dsz%0.2dst%03d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0]);
       status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

       status = H5Gget_objinfo (file_id, tagname, 0, NULL);
   
       if (status != 0){

         /* Create a group named "/MyGroup" in the file. */
         group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

         /* Close the group. */
         status = H5Gclose(group_id);

       }
       snprintf(tagname,400,"/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2]);
       status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

       status = H5Gget_objinfo (file_id, tagname, 0, NULL);

       if (status != 0){

         /* Create a group named "/MyGroup" in the file. */
         group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

         /* Close the group. */
         status = H5Gclose(group_id);

       }



       for ( int ibeta = 0; ibeta < irrep_dim; ibeta++ ) {


         snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta );

         status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

         status = H5Gget_objinfo (file_id, tagname, 0, NULL);
   
         if (status != 0){

           /* Create a group named "/MyGroup" in the file. */
           group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

           /* Close the group. */
           status = H5Gclose(group_id);

         } 

         for ( int imu = 0; imu < irrep_dim; imu++ ) {


           snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d",source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu );

           status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

           status = H5Gget_objinfo (file_id, tagname, 0, NULL);
           if (status != 0){

             /* Create a group named "/MyGroup" in the file. */
             group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

             /* Close the group. */
             status = H5Gclose(group_id);

           }
 
           for ( int icombination_source=0; icombination_source < g_twopoint_function_list[i2pt].ncombination_total_momentum_source ; ++icombination_source ) {
             snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source] );

             status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

             status = H5Gget_objinfo (file_id, tagname, 0, NULL);
             if (status != 0){

               /* Create a group named "/MyGroup" in the file. */
               group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

               /* Close the group. */
               status = H5Gclose(group_id);

             }    

             /***********************************************************
              * read data block from h5 file : source
              ***********************************************************/

             snprintf ( filename, 400, "projection_coefficients_%s_group_%s_irrep_%s.h5",
                            g_twopoint_function_list[i2pt].particlename_source,
                            g_twopoint_function_list[i2pt].group,
                            g_twopoint_function_list[i2pt].irrep );
             if (strcmp( g_twopoint_function_list[i2pt].particlename_source, "piN") == 0){
                snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], imu, ibeta );
             }
             else {
                snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2], imu, ibeta );
             }

             int *Nps_Nreplica_source= init_1level_itable( 3 );


             exitstatus = read_from_h5_file ( (void*)(Nps_Nreplica_source), filename, tagname, io_proc, 1 );
             if (exitstatus != 0){
               fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
               exit(1); 
             }


             int ** momtable_source = init_2level_itable ( Nps_Nreplica_source[1], 6 );
             if (strcmp ( g_twopoint_function_list[i2pt].particlename_source, "piN")==0){
               snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], imu, ibeta );
             }
             else {
               snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1], Ptot[2], imu, ibeta );
             }


             exitstatus = read_from_h5_file ( (void*)(momtable_source[0]), filename, tagname, io_proc, 1 );
             if (exitstatus != 0){
               fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
               exit(1); 
             }
             double ***projection_coeff_c_ORT= init_3level_dtable( Nps_Nreplica_source[1]*spin1212dimension*spin1dimension_source, Nps_Nreplica_source[0], 2);
             if (strcmp ( g_twopoint_function_list[i2pt].particlename_source, "piN")==0){
               snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/c_data_ort",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], imu, ibeta );
             } else {
               snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/c_data_ort",  Ptot[0],Ptot[1], Ptot[2], imu, ibeta );
             }
             fprintf(stdout, "# [piN2piN_projection_apply] projection coeff source loaded\n");

             exitstatus = read_from_h5_file ( (void*)(projection_coeff_c_ORT[0][0]), filename, tagname, io_proc, 0 );
             if (exitstatus != 0){
               fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
               exit(1);
             }

             for ( int icombination_sink=0; icombination_sink < g_twopoint_function_list[i2pt].ncombination_total_momentum_sink ; ++icombination_sink ) {

               snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink]  );

               status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

               status = H5Gget_objinfo (file_id, tagname, 0, NULL);
               if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

               }



               /***********************************************************
                * read data block from h5 file : sink
                ***********************************************************/

               snprintf ( filename, 400, "projection_coefficients_%s_group_%s_irrep_%s.h5",
                            g_twopoint_function_list[i2pt].particlename_sink,
                            g_twopoint_function_list[i2pt].group,
                            g_twopoint_function_list[i2pt].irrep );

               if (strcmp(g_twopoint_function_list[i2pt].particlename_sink , "piN" )==0) {
                  snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], imu, ibeta );
               } 
               else {
                  snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1], Ptot[2],  imu, ibeta );
               }

               int *Nps_Nreplica_sink= init_1level_itable( 3 );


               exitstatus = read_from_h5_file ( (void*)(Nps_Nreplica_sink), filename, tagname, io_proc, 1 );
               if (exitstatus != 0){
                 fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
                 exit(1); 
               }

               if ( strcmp(  g_twopoint_function_list[i2pt].particlename_sink, "piN" ) == 0){
                 snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], imu, ibeta );
               } else {
                 snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1], Ptot[2], imu, ibeta );
               }

               int ** momtable_sink = init_2level_itable ( Nps_Nreplica_sink[1], 6 );
 

               exitstatus = read_from_h5_file ( (void*)(momtable_sink[0]), filename, tagname, io_proc, 1 );
               if (exitstatus != 0){
                 fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
                 exit(1); 
               }


               double ***projection_coeff_a_ORT= init_3level_dtable(Nps_Nreplica_sink[0], Nps_Nreplica_sink[1]*spin1212dimension*spin1dimension_sink,2);

               if (strcmp(g_twopoint_function_list[i2pt].particlename_sink , "piN" )==0) {
                  snprintf( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/a_data_ort",  Ptot[0],Ptot[1], Ptot[2], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], imu, ibeta );
               } else {
                  snprintf( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/a_data_ort",  Ptot[0],Ptot[1], Ptot[2],  imu, ibeta );

               }

               exitstatus = read_from_h5_file ( (void*)(projection_coeff_a_ORT[0][0]), filename, tagname, io_proc, 0 );
               if (exitstatus != 0){
                 fprintf(stderr, "# [piN2piN_projection_apply] Error in opening %s %s\n",filename, tagname );
                 exit(1);
               }



               for (int gamma_nplettid_source=0; gamma_nplettid_source < numberofnplets_source; ++gamma_nplettid_source) {

                 int *gamma_table_source=init_1level_itable(spin1dimension_source);
                 for (int iii=0; iii<spin1dimension_source;++iii){
                   gamma_table_source[iii]=g_twopoint_function_list[i2pt].list_of_gammas_i1[gamma_nplettid_source*spin1dimension_source+iii][0];
                 }
                 char *gamma_string_source=tagname_forgamma_multiplets ( gamma_table_source, spin1dimension_source );

                 snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d/source_%s/",source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], gamma_string_source);

                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                   /* Create a group named "/MyGroup" in the file. */
                   group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                   /* Close the group. */
                   status = H5Gclose(group_id);

                 }

                 for (int gamma_nplettid_sink=0; gamma_nplettid_sink < numberofnplets_sink; ++gamma_nplettid_sink) {

                   int *gamma_table_sink=init_1level_itable(spin1dimension_sink);
                   for (int iii=0; iii<spin1dimension_sink;++iii){
                     gamma_table_sink[iii]=g_twopoint_function_list[i2pt].list_of_gammas_f1[gamma_nplettid_sink*spin1dimension_sink+iii][0];
                   }
                   char *gamma_string_sink=tagname_forgamma_multiplets ( gamma_table_sink, spin1dimension_sink );

                   snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d/source_%s/sink_%s",source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], gamma_string_source, gamma_string_sink);

                   status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                   status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                   if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                   }

                   for (int nreplicum_source=0; nreplicum_source < Nps_Nreplica_source[0] ; ++nreplicum_source ) {

                     snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d/source_%s/sink_%s/Replicasource_%d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], gamma_string_source, gamma_string_sink, nreplicum_source);

                     status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                     status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                     if (status != 0){

                       /* Create a group named "/MyGroup" in the file. */
                       group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                       /* Close the group. */
                       status = H5Gclose(group_id);

                     }

 
                     for ( int nreplicum_sink=0; nreplicum_sink < Nps_Nreplica_sink[0] ; ++nreplicum_sink ) {

                       snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d/source_%s/sink_%s/Replicasource_%d/Replicasink_%d", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], gamma_string_source, gamma_string_sink, nreplicum_source, nreplicum_sink);

                       status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                       status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                       if (status != 0){
 
                         /* Create a group named "/MyGroup" in the file. */
                         group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                         /* Close the group. */
                         status = H5Gclose(group_id);

                       }
 
                       double **projected_correlation_function=init_2level_dtable ( g_twopoint_function_list[i2pt].T, 2);

                       for ( int i1_source =0; i1_source < Nps_Nreplica_source[1] ; ++i1_source ) {
      
                         for ( int i2_source=0; i2_source < spin1dimension_source ; ++i2_source ) {

                           char *gamma_string_source_member=convert_gamma_to_string( gamma_table_source[i2_source]);
                           printf("Plettid %d spin1 %d i2source %d gamma %d\n", gamma_nplettid_source,spin1dimension_source, i2_source,gamma_table_source[i2_source]);


                           for ( int i1_sink =0; i1_sink < Nps_Nreplica_sink[1] ; ++i1_sink ) {

                             for ( int i2_sink =0; i2_sink < spin1dimension_sink ; ++i2_sink ) {
                               printf("Plettid %d spin1 %d i2sink %d gamma %d\n", gamma_nplettid_sink,spin1dimension_sink, i2_sink,gamma_table_sink[i2_sink]);

                               char *gamma_string_sink_member=convert_gamma_to_string( gamma_table_sink[i2_sink]);


                               int pf1x=momtable_sink[i1_sink][0];
                               int pf1y=momtable_sink[i1_sink][1];
                               int pf1z=momtable_sink[i1_sink][2];
                               int pf2x=momtable_sink[i1_sink][3];
                               int pf2y=momtable_sink[i1_sink][4];
                               int pf2z=momtable_sink[i1_sink][5];

                               int pi1x=momtable_source[i1_source][0];
                               int pi1y=momtable_source[i1_source][1];
                               int pi1z=momtable_source[i1_source][2];
                               int pi2x=momtable_source[i1_source][3];
                               int pi2y=momtable_source[i1_source][4];
                               int pi2z=momtable_source[i1_source][5];

                               char *correlation_function_filename_suffix=(char *)malloc(sizeof(char)*100);
                               char *correlation_function_tagname_suffix =(char *)malloc(sizeof(char)*100);

                               if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "D") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "D") == 0) ){

                                snprintf(correlation_function_filename_suffix, 100, "D");
                                snprintf(correlation_function_tagname_suffix, 100, "D"); 

                               } else if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "D") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "piN") == 0 )){

                                snprintf(correlation_function_filename_suffix, 100, "T");
                                snprintf(correlation_function_tagname_suffix, 100, "T"); 

                               } else if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "piN") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "D") == 0)){

                                snprintf(correlation_function_filename_suffix, 100, "TpiNsink");
                                snprintf(correlation_function_tagname_suffix, 100, "TpiNsink");

                               } else if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "piN") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "piN") == 0)){

                                snprintf(correlation_function_filename_suffix, 100, "piN");
                                snprintf(correlation_function_tagname_suffix, 100, "piN");

                               } else if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "N") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "N") == 0)){

                                snprintf(correlation_function_filename_suffix, 100, "N");
                                snprintf(correlation_function_tagname_suffix, 100, "N");

                               }
                               else { 
                                fprintf(stderr, "#[piN2piN_projection_apply] only piN -delta system is implemented\n");
                                exit(1);
                               }

 
                               if ((strcmp(g_twopoint_function_list[i2pt].particlename_sink, "piN") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "piN") == 0)){
 
                                  snprintf ( tagname, 400, "/sx%.02dsy%.02dsz%.02dst%03d/gf25/pf2x%.02dpf2y%.02dpf2z%.02d/gf1%s/pf1x%.02dpf1y%.02dpf1z%.02d/gi25/pi2x%.02dpi2y%.02dpi2z%.02d/gi1%s/pi1x%.02dpi1y%.02dpi1z%.02d/%s", source_coords_list[k][1],
                                                        source_coords_list[k][2],
                                                        source_coords_list[k][3],
                                                        source_coords_list[k][0],
                                                        pf2x,
                                                        pf2y,
                                                        pf2z,
                                                        gamma_string_sink_member,
                                                        pf1x,
                                                        pf1y,
                                                        pf1z,
                                                        pi2x,
                                                        pi2y,
                                                        pi2z,
                                                        gamma_string_source_member,
                                                        pi1x,
                                                        pi1y,
                                                        pi1z,
                                                        correlation_function_tagname_suffix);
                               } else if ((strcmp(g_twopoint_function_list[i2pt].particlename_sink, "N") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "N") == 0) || ((strcmp(g_twopoint_function_list[i2pt].particlename_sink, "D") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "D") == 0))){
                                 snprintf ( tagname, 400, "/sx%.02dsy%.02dsz%.02dst%03d/gf1%s/pf1x%.02dpf1y%.02dpf1z%.02d/gi1%s/pi1x%.02dpi1y%.02dpi1z%.02d/%s", source_coords_list[k][1],
                                                        source_coords_list[k][2],
                                                        source_coords_list[k][3],
                                                        source_coords_list[k][0],
                                                        gamma_string_sink_member,
                                                        pf1x,
                                                        pf1y,
                                                        pf1z,
                                                        gamma_string_source_member,
                                                        pi1x,
                                                        pi1y,
                                                        pi1z,
                                                        correlation_function_tagname_suffix);
 
                               } else  if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "D") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "piN") == 0 )){
                                 snprintf ( tagname, 400, "/sx%.02dsy%.02dsz%.02dst%03d/gf1%s/pf1x%.02dpf1y%.02dpf1z%.02d/gi25/pi2x%.02dpi2y%.02dpi2z%.02d/gi1%s/pi1x%.02dpi1y%.02dpi1z%.02d/%s", source_coords_list[k][1],
                                                        source_coords_list[k][2],
                                                        source_coords_list[k][3],
                                                        source_coords_list[k][0],
                                                        gamma_string_sink_member,
                                                        pf1x,
                                                        pf1y,
                                                        pf1z,
                                                        pi2x,
                                                        pi2y,
                                                        pi2z,
                                                        gamma_string_source_member,
                                                        pi1x,
                                                        pi1y,
                                                        pi1z,
                                                        correlation_function_tagname_suffix);
                               } else if ( (strcmp(g_twopoint_function_list[i2pt].particlename_sink, "piN") == 0) && (strcmp(g_twopoint_function_list[i2pt].particlename_source, "D") == 0) ){
                                 snprintf ( tagname, 400, "/sx%.02dsy%.02dsz%.02dst%03d/gf25/pf2x%.02dpf2y%.02dpf2z%.02d/gf1%s/pf1x%.02dpf1y%.02dpf1z%.02d/gi1%s/pi1x%.02dpi1y%.02dpi1z%.02d/%s", source_coords_list[k][1],
                                                        source_coords_list[k][2],
                                                        source_coords_list[k][3],
                                                        source_coords_list[k][0],
                                                        pf2x,
                                                        pf2y,
                                                        pf2z,
                                                        gamma_string_sink_member,
                                                        pf1x,
                                                        pf1y,
                                                        pf1z,
                                                        gamma_string_source_member,
                                                        pi1x,
                                                        pi1y,
                                                        pi1z,
                                                        correlation_function_tagname_suffix);

                                 
                               } else {
                                  fprintf(stderr, "Projector for particle at sink %s and at source %s is not implemented yet",g_twopoint_function_list[i2pt].particlename_sink,g_twopoint_function_list[i2pt].particlename_source);
                                  exit(1);
                               }

                               const int i_total_momentum=Ptot[0]*Ptot[0]+Ptot[1]*Ptot[1]+Ptot[2]*Ptot[2];
                             
                               snprintf ( filename, 400, "%s%04d_PX%.02dPY%.02dPZ%.02d_%s.h5",
                                                        filename_prefix,
                                                        Nconf,
                                                        momentum_orbit_pref[i_total_momentum][0],
                                                        momentum_orbit_pref[i_total_momentum][1],
                                                        momentum_orbit_pref[i_total_momentum][2],
                                                        correlation_function_filename_suffix);
                               printf("Filename=%s\n", filename);
                               double ***correlation_function=init_3level_dtable( g_twopoint_function_list[i2pt].T,g_twopoint_function_list[i2pt].d*g_twopoint_function_list[i2pt].d ,2 );

                               exitstatus = read_from_h5_file ( (void*)(correlation_function[0][0]), filename, tagname, io_proc, 0 );

                               for ( int i3_source =0; i3_source < spin1212dimension; ++i3_source ) {

                                 for ( int i3_sink =0; i3_sink < spin1212dimension ; ++i3_sink ) {

                                   for (int t=0; t<g_twopoint_function_list[i2pt].T; ++t){

                                     projected_correlation_function[t][0]+=
                                       +projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][0]*correlation_function[t][i3_source*4+i3_sink][0]
                                       -projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][1]*correlation_function[t][i3_source*4+i3_sink][0]
                                       -projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][0]*correlation_function[t][i3_source*4+i3_sink][1]
                                       -projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][1]*correlation_function[t][i3_source*4+i3_sink][1];

                                     projected_correlation_function[t][1]+=
                                       -projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][1]*correlation_function[t][i3_source*4+i3_sink][1]
                                       +projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][0]*correlation_function[t][i3_source*4+i3_sink][1]
                                       +projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][0]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][1]*correlation_function[t][i3_source*4+i3_sink][0]
                                       +projection_coeff_a_ORT[nreplicum_sink][i1_sink*spin1dimension_sink*spin1212dimension+i2_sink*spin1212dimension+i3_sink][1]*projection_coeff_c_ORT[i1_source*spin1dimension_source*spin1212dimension+i2_source*spin1212dimension+i3_source][nreplicum_source][0]*correlation_function[t][i3_source*4+i3_sink][0];

                                   }/*time*/

                                 }/*i3 sink */

                               }/*i3_source */


                               fini_3level_dtable(&correlation_function);

                               free(correlation_function_filename_suffix);
     
                               free(correlation_function_tagname_suffix);
  
                               free(gamma_string_sink_member);

                             }/*i2_sink */

                           }/*i1_sink */

                           free(gamma_string_source_member);

                         }/*i2_source*/

                       }/*i1_source*/

                       snprintf ( tagname, 400, "/sx%0.2dsy%0.2dsz%0.2dst%03d/px%dpy%dpz%d/beta_%d/mu_%d/source_totmomN%dp%d/sink_totmomN%dp%d/source_%s/sink_%s/Replicasource_%d/Replicasink_%d/data", source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         source_coords_list[0][0], Ptot[0],Ptot[1],Ptot[2], ibeta, imu, g_twopoint_function_list[i2pt].total_momentum_nucleon_source[icombination_source], g_twopoint_function_list[i2pt].total_momentum_pion_source[icombination_source],g_twopoint_function_list[i2pt].total_momentum_nucleon_sink[icombination_sink], g_twopoint_function_list[i2pt].total_momentum_pion_sink[icombination_sink], gamma_string_source, gamma_string_sink, nreplicum_source, nreplicum_sink);

                       hsize_t dims[2];
                       dims[0]=g_twopoint_function_list[i2pt].T;
                       dims[1]=2;
                       dataspace_id = H5Screate_simple(2, dims, NULL);


                       /* Create a dataset in group "MyGroup". */
                       dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                       /* Write the first dataset. */
                       status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(projected_correlation_function[0][0]));

                       /* Close the data space for the first dataset. */
                       status = H5Sclose(dataspace_id);

                       /* Close the first dataset. */
                       status = H5Dclose(dataset_id);

                       fini_2level_dtable(&projected_correlation_function);

                     } /*nreplicum sink */


                   } /*nreplicum source */

                   fini_1level_itable(&gamma_table_sink);
                   free(gamma_string_sink);


                 } /*ngamma multiplett sink*/

                 fini_1level_itable(&gamma_table_source);
                 free(gamma_string_source);

               } /*ngamma muplitplett source*/

               fini_1level_itable(&Nps_Nreplica_sink);
               fini_2level_itable(&momtable_sink);
               fini_3level_dtable(&projection_coeff_a_ORT);


             } /* momentum combination sink */

             fini_1level_itable(&Nps_Nreplica_source);
             fini_2level_itable(&momtable_source);
             fini_3level_dtable(&projection_coeff_c_ORT);
      
           } /* momentum combination source */

         } /*beta */

       } /*mu*/ 


     }  // end of loop on 2-point functions

   } //end of loop on source positions 

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
