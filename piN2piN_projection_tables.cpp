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

double _Complex ** apply_gramschmidt( double _Complex ** Projector, int annihilation_or_creation, int *dimension) {
   
   int i0;
   int final_dimension=0;
   double _Complex ** orthogonalized=init_2level_ztable(*dimension, *dimension);

   if (annihilation_or_creation == ANNIHILATION){

    /* Looking for the first non-zero row */
     int check=1;
     for (i0=0;i0<*dimension ;++i0){
       check=1;
       for (int j=0; j<*dimension;++j){
         if (cabs(Projector[i0][j]) > EPS ){
          check=0;
          break;
         } 
       }
       if (check==0){
        break;
       }     
     }
     if (check==1){
       fprintf(stderr, "# [apply_gramschmidt] Error the whole matrix is zero appearently the irrep does not contribute\n" );
       *dimension=0;
       return NULL;
     }
     /* Normalizing the first vector */
     double norm=0.0;
     for (int j=0; j<*dimension; ++j){
       norm+=creal(Projector[i0][j])*creal(Projector[i0][j])+cimag(Projector[i0][j])*cimag(Projector[i0][j]);
     }
     for (int j=0; j<*dimension; ++j){
       orthogonalized[0][j]=1./sqrt(norm)*Projector[i0][j];
     }
     if (i0==(*dimension-1)){
       *dimension=i0; 
       return  orthogonalized;
     }
     /* Orthogonalizing the remaining ones*/
     for (int ii=i0+1; ii<*dimension; ++ii){
       check=1;
       /* Looking for the next non-zero row */
       for (int j=0; j<*dimension;++j){
         if (cabs(Projector[ii][j]) > EPS ){
          check=0;
          break;
         }
       }
       /* If check is zero we found a non-zero row, lets ortogonalize it to all the previous ones*/
       if (check==1){
         continue;
       }
       for (int j=0; j< *dimension; ++j)
         orthogonalized[final_dimension+1][j]=Projector[ii][j];
       for (int iib=0; iib<=final_dimension; ++iib){
         double _Complex prodij=scalar_product( orthogonalized[iib], Projector[ii], *dimension );
         for (int j=0; j< *dimension; ++j)
           orthogonalized[final_dimension+1][j]-=prodij*orthogonalized[iib][j];
       }
       norm=0.;
       for (int j=0; j<*dimension; ++j){
         norm+= creal(orthogonalized[final_dimension+1][j])*creal(orthogonalized[final_dimension+1][j])+cimag(orthogonalized[final_dimension+1][j])*cimag(orthogonalized[final_dimension+1][j]);
       }
       if (norm<EPS)
         continue;
       else{
         for (int j=0; j<*dimension; ++j){
           orthogonalized[final_dimension+1][j]=1./sqrt(norm)*orthogonalized[final_dimension+1][j];
         }
         final_dimension+=1;
       }
     }
   }
   else if (annihilation_or_creation == CREATION){
     /* Looking for the first non-zero column */
     int check=1;
     for (i0=0;i0<*dimension ;++i0){
       check=1;
       for (int j=0; j<*dimension;++j){
         if (cabs(Projector[j][i0]) > EPS ){
          check=0;
          break;
         }
       }
       if (check==0){
        break;
       }
     }
     if (check==1){
       fprintf(stderr, "# [apply_gramschmidt] Error the whole matrix is zero appearently the irrep does not contribute\n" );
       *dimension=0;
       return NULL;
     }
     /* Normalizing the first vector */
     double norm=0.0;
     for (int j=0; j<*dimension; ++j){
       norm+=creal(Projector[j][i0])*creal(Projector[j][i0])+cimag(Projector[j][i0])*cimag(Projector[j][i0]);
     }
     for (int j=0; j<*dimension; ++j){
       orthogonalized[j][0]=1./sqrt(norm)*Projector[j][i0];
     }
     if (i0==(*dimension-1)){
       *dimension=i0;
       return  orthogonalized;
     }
     /* Orthogonalizing the remaining ones*/
     for (int ii=i0+1; ii<*dimension; ++ii){
       check=1;
       /* Looking for the next non-zero row */
       for (int j=0; j<*dimension;++j){
         if (cabs(Projector[j][ii]) > EPS ){
          check=0;
          break;
         }
       }
       /* If check is zero we found a non-zero row, lets ortogonalize it to all the previous ones*/
       if (check==1){
         continue;
       }
       for (int j=0; j< *dimension; ++j)
         orthogonalized[j][final_dimension+1]=Projector[j][ii];

       double _Complex  *ortho2vec=init_1level_ztable( *dimension );
       for (int iic=0; iic< *dimension; ++iic){
         ortho2vec[iic]=Projector[iic][ii];
       }
       for (int iib=0; iib<=final_dimension; ++iib){
         double _Complex  *ortho1vec=init_1level_ztable( *dimension );
         for (int iic=0; iic < *dimension; ++iic){
           ortho1vec[iic]=orthogonalized[iic][iib];
         }
         double _Complex prodij=scalar_product( ortho1vec, ortho2vec, *dimension );
         for (int j=0; j< *dimension; ++j)
           orthogonalized[j][final_dimension+1]-=prodij*orthogonalized[j][iib];
         fini_1level_ztable( &ortho1vec );
       }
       fini_1level_ztable( &ortho2vec );
       norm=0.;
       for (int j=0; j<*dimension; ++j){
         norm+= creal(orthogonalized[j][final_dimension+1])*creal(orthogonalized[j][final_dimension+1])+cimag(orthogonalized[j][final_dimension+1])*cimag(orthogonalized[j][final_dimension+1]);
       }
       if (norm<EPS)
         continue;
       else{
         for (int j=0; j<*dimension; ++j){
           orthogonalized[j][final_dimension+1]=1./sqrt(norm)*orthogonalized[j][final_dimension+1];
         }
         final_dimension+=1;
       }
     }
   }
    
   *dimension=final_dimension+1;
   return(orthogonalized);
}


/***********************************************************
 * main program
 * This code is supposed to be produce table of coefficients
 * to form maximal number of linearly independent interpolating
 * operators for the D-D, N-N, piN-piN case
 ***********************************************************/
int main(int argc, char **argv) {
 
#define _ZCOEFF_EPS 8.e-12

#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover; */
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover; */
#endif


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
   * set Cg basis projection coefficients
   ***********************************************************/
  double const Cgamma_basis_matching_coeff[16] = {
   -1.00,  /*  0 =  Cgy       g0*-i=cgy   */
   -1.00,  /*  1 =  Cgzg5     g1*-i=cgzg5 */
   +1.00,  /*  2 =  Cg0       g2*+i=cg0   */
   +1.00,  /*  3 =  Cgxg5     g3*+i=cgxg5 */
   -1.00,  /*  4 =  Cgyg0     g4*-i=cgyg0 */
   +1.00,  /*  5 =  Cgyg5g0   g5*+i=cgyg5g0 */
   -1.00,  /*  6 =  Cgyg5     g6*-i=cgyg5   */
   -1.00,  /*  7 =  Cgz       g7*-i=cgz */
   -1.00,  /*  8 =  Cg5g0     g8*-i=cg5g0 */
   +1.00,  /*  9 =  Cgx       g9*+i=cgx */
   +1.00,  /* 10 =  Cgzg5g0  g10*+i=cgzg5g0  */
   -1.00,  /* 11 =  C        g11*-i=C  */
   -1.00,  /* 12 =  Cgxg5g0  g12*-i=Cgxg5g0  */
   +1.00,  /* 13 =  Cgxg0    g13*+i=cgxg0  */
   +1.00,  /* 14 =  Cg5      g14*+i=Cg5 */
   -1.00   /* 15 =  Cgzg0    g15*-i=cgzg0*/
  };

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

    printf("# [piN2piN_projection_table] start analyzing twopoint function index %d\n", i2pt);

    for (int ii=0; ii< g_twopoint_function_list[i2pt].nlistmomentumf1; ++ii){

      printf("# [piN2piN_projection_table] pf1 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf1list[ii][0] , g_twopoint_function_list[i2pt].pf1list[ii][1] , g_twopoint_function_list[i2pt].pf1list[ii][2] );
      printf("# [piN2piN_projection_table] pf2 (%d %d %d)\n", g_twopoint_function_list[i2pt].pf2list[ii][0] , g_twopoint_function_list[i2pt].pf2list[ii][1] , g_twopoint_function_list[i2pt].pf2list[ii][2] );

    }


    /******************************************************
     * print the 2-point function parameters
     ******************************************************/
    sprintf ( filename, "twopoint_function_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection_table] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "TWPT", ofs );
    fclose ( ofs );



    /****************************************************
     * read little group parameters
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


    int const nrot      = projector.rtarget->n;
    int const irrep_dim = projector.rtarget->dim;


    /******************************************************/
    /******************************************************/
    /******************************************************
     *   The dimension of the irrep matrix
     ******************************************************/
    int const n_tp_project = irrep_dim * irrep_dim ;

    /******************************************************
     * loop on elements of tp_project
     * - initialize
     * - copy content of current reference element of
     *   g_twopoint_function_list
     ******************************************************/

    const int spin1dimension = g_twopoint_function_list[i2pt].number_of_gammas_f1;
    const int spin1212dimension = g_twopoint_function_list[i2pt].d;
    const int momentumlistsize = g_twopoint_function_list[i2pt].nlistmomentumf1 ;
    const int dimension_coeff = g_twopoint_function_list[i2pt].number_of_gammas_f1*g_twopoint_function_list[i2pt].d*momentumlistsize; 

    gamma_matrix_type gl, gr, gf11, gf12, gf2, gi11, gi12,gi2;
    gamma_matrix_init ( &gl );
    gamma_matrix_init ( &gr );


    for ( int ibeta = 0; ibeta < irrep_dim; ibeta++ ) {

      for ( int imu = 0; imu < irrep_dim; imu++ ) {


        double _Complex ** projection_matrix_a = init_2level_ztable (dimension_coeff , dimension_coeff);
        if ( projection_matrix_a == NULL ) {
          fprintf(stderr,"[piN2piN_projection_table] Error in allocating matrix for P\n");
        }

        double _Complex ** projection_matrix_c = init_2level_ztable (dimension_coeff , dimension_coeff);
        if ( projection_matrix_c == NULL ) {
          fprintf(stderr,"[piN2piN_projection_table] Error in allocating matrix for P\n");
        }

        /******************************************************
         * curren projection coefficient for chosen irrep rows
         * at source and sink, together with sign factors
         * from rotation+basis projection of gamma matrices
         ******************************************************/

        /******************************************************
         *  loop on all the spin1 degrees of freedom
         * 
         *  for the delta cgx,cgy,cgz
         *  for the nucleon cg5
         *
         *****************************************************/
        for (int spin1degree=0; spin1degree < spin1dimension ; ++ spin1degree ){

          /*********************************************************************  
           *  loop on all the spin1212 degrees of freedom
           *
           *  this loop is on the internal spin indices of the nucleon or delta
           ********************************************************************/
  
          for (int spin1212degree = 0; spin1212degree < spin1212dimension; ++spin1212degree ) { 

            /*******************************************************
             *  loop on the possible combination of pf1 and pf2 
             *  giving the same total momentum and relativ momentum 
             *******************************************************/

            for ( int momentum_idx=0; momentum_idx < momentumlistsize ; ++momentum_idx ){


              /******************************************************
               * loop on little group elements
               *  - rotations and rotation-reflections
               *  - at sink, left-applied element
               ******************************************************/

              for ( int irotl = 0; irotl < 2*nrot; irotl ++ ) {


                double _Complex *spin1212_vector=init_1level_ztable(spin1212dimension);
                double _Complex *spin1212_rotated_vector=init_1level_ztable(spin1212dimension);

                spin1212_vector[spin1212degree]=1.;


                if ( g_verbose > 3 ) fprintf ( stdout, "# [piN2piN_projection] left rotref %2d - %2d\n", irotl / nrot, irotl % nrot );

                /******************************************************
                 * set momentum vector ( = spin-1 ) rotation matrix
                 *
                 *                                          proper rotation          rotation-reflection
                 ******************************************************/
                double _Complex ** Rpl = ( irotl < nrot ) ? projector.rp->R[irotl] : projector.rp->IR[irotl-nrot];

                /******************************************************
                 * rotate the final momentum vectors pf1, pf2 with Rpl
                 * left = sink side
                 ******************************************************/
                int pf1[3];
                int pf2[3];
                rot_point ( pf1, g_twopoint_function_list[i2pt].pf1list[momentum_idx], Rpl );
                rot_point ( pf2, g_twopoint_function_list[i2pt].pf2list[momentum_idx], Rpl );

                /******************************************************
                 * Determining the index of the momentum combination
                 *
                 ******************************************************/
                 
                int check=0;
                int rotated_momentum_index=0;
                for (int ii=0; ii< momentumlistsize; ++ii){
                  if ( ( pf1[0]== g_twopoint_function_list[i2pt].pf1list[ii][0] ) &&
                       ( pf1[1]== g_twopoint_function_list[i2pt].pf1list[ii][1] ) &&
                       ( pf1[2]== g_twopoint_function_list[i2pt].pf1list[ii][2] ) &&
                       ( pf2[0]== g_twopoint_function_list[i2pt].pf2list[ii][0] ) &&
                       ( pf2[1]== g_twopoint_function_list[i2pt].pf2list[ii][1] ) &&
                       ( pf2[2]== g_twopoint_function_list[i2pt].pf2list[ii][2] )  ){
                    check=1;
                    rotated_momentum_index=ii;
                    break;
                  }
                }
                if (check == 0){
                  fprintf(stderr, "[piN2piN_projection_tables] Error some rotation is problematic\n");
                  exit(1);
                }
             

                /******************************************************
                 * set the spinor ( = spin-1/2 bispinor ) rotation matrix
                 *
                 *                                          proper rotation               rotation-reflection
                 ******************************************************/
                double _Complex ** Rsl = ( irotl < nrot ) ? projector.rspin[0].R[irotl] : projector.rspin[0].IR[irotl-nrot];
          
                memcpy ( gl.v, Rsl[0], 16*sizeof(double _Complex) );

                if ( g_verbose > 4 ) gamma_matrix_printf ( &gl, "gl", stdout ); 

                /******************************************************
                 * Gamma_{f_1, 1/2} --->
                 *
                 *   S(R)^* Gamma_{f_1, 1/2} S(R)^H
                 *
                 *   ****************************
                 *   * ANNIHILATION / SINK SIDE *
                 *   ****************************
                 ******************************************************/
                /* set and rotate gf11 */
                gamma_matrix_set ( &gf11, g_twopoint_function_list[i2pt].list_of_gammas_f1[spin1degree][0], Cgamma_basis_matching_coeff[g_twopoint_function_list[i2pt].list_of_gammas_f1[spin1degree][0]] );
                /* gl^C gf11 gl^H */
                gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf11, &gl, 'C', &gf11, &gl, 'H' );

                /* set and rotate gf12 */
                gamma_matrix_set ( &gf12, g_twopoint_function_list[i2pt].list_of_gammas_f1[spin1degree][1], 1. );
                /* gl^N gf12 gl^H */
                gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf12, &gl, 'N', &gf12, &gl, 'H' );


                /* check for not set or error */
              
                int rotated_gamma_id =  -1;
                for (int ii=0; ii < g_twopoint_function_list[i2pt].number_of_gammas_f1; ++ii ) {
                  if (gf11.id == g_twopoint_function_list[i2pt].list_of_gammas_f1[ii][0]) {
                    rotated_gamma_id = ii; 
                    break;
                  }
                }
                if (rotated_gamma_id == -1){ 
                  fprintf ( stderr, "[piN2piN_projection_table] Error from gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op for gf1 %d irotl %d gf11.id %s %d\n",irotl,gf11.id,  __FILE__, __LINE__ );
                  EXIT(217);
                }

                /******************************************************
                 * correct gf11 for basis matching
                 ******************************************************/
                gf11.s *= Cgamma_basis_matching_coeff[ gf11.id ];

                /* transcribe gf1 gamma ids to tp.gf1 */
                int gf1[2];
                gf1[0] = gf11.id;
                gf1[1] = gf12.id;

                /******************************************************
                 * Gamma_{f_2} ---> S(R) Gamma_{f_2} S(R)^+
                 ******************************************************/
                /* set and rotate gf2 */
                gamma_matrix_set ( &gf2, g_twopoint_function_list[i2pt].list_of_gammas_f2[0], 1. );
                /* gl^N gf2 gl^H */
                gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf2, &gl, 'N', &gf2, &gl, 'H' );

                /* transcribe gf2 gamma id to tp.gf2 */
                int gf2_local;
                gf2_local    = gf2.id;

                rot_mat_adj_ti_vec ( spin1212_rotated_vector, Rsl, spin1212_vector, spin1212dimension );
               
                /******************************************************
                 * projection variants
                 ******************************************************/

                /******************************************************
                 * irrep matrix for left-applied rotation
                 *   = sink side
                 ******************************************************/
                double _Complex ** Tirrepl = ( irotl < nrot ) ? projector.rtarget->R[irotl] : projector.rtarget->IR[irotl-nrot];



                double _Complex const zcoeff = 
                          (double)( projector.rtarget->dim  ) /( 2. *    projector.rtarget->n     )                          
                            * gf11.s                               /* gamma rot sign f_1,1      */
                            * gf12.s                               /* gamma rot sign f_1,2      */
                            * gf2.s                                /* gamma rot sign f_2        */
                            * Tirrepl[imu][ibeta];           /* phase irrep matrix sink   */;
                if ( cabs( zcoeff ) < _ZCOEFF_EPS ) {
                /* if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] | zcoeff = %16.7e + I %16.7e | < _ZCOEFF_EPS, continue\n", creal( zcoeff ), cimag( zcoeff ) ); */
                    continue;
                }
                if ( g_verbose > 4 ){
                  char * text_output=(char *)malloc(sizeof(char)*100);
                  snprintf(text_output,100,"Rvector(sink_row=%d,ref_sink=%d)[(spin1=%d,spi1212=%d,irotl=%d)]",imu, ibeta,spin1degree, spin1212degree, irotl, imu,ibeta);

                 rot_printf_vec (spin1212_rotated_vector,4,text_output, stdout );
                 free(text_output);
                }

                rot_vec_pl_eq_rot_vec_ti_co ( &projection_matrix_a[momentum_idx*spin1dimension*spin1212dimension+spin1degree*spin1212dimension + spin1212degree][rotated_momentum_index*spin1212dimension*spin1dimension+spin1212dimension*rotated_gamma_id], spin1212_rotated_vector , zcoeff, spin1212dimension);

                fini_1level_ztable(&spin1212_vector);
 
                fini_1level_ztable(&spin1212_rotated_vector);


              } /* end of loop over rotation elements */

            } /* end of loop over different momentum combination */
          
          } /* end of loop over spi1212degree */

        } /* end of loop over spin1degree */

        snprintf ( filename, 400, "projection_coefficients_%s_group_%s_irrep_%s.h5",
                         g_twopoint_function_list[i2pt].name, projector.rtarget->group , projector.rtarget->irrep);
        hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
        herr_t      status;
        int pfnx=g_twopoint_function_list[i2pt].pf1list[0][0];
        int pfny=g_twopoint_function_list[i2pt].pf1list[0][1];
        int pfnz=g_twopoint_function_list[i2pt].pf1list[0][2];

        int pfpx=g_twopoint_function_list[i2pt].pf2list[0][0];
        int pfpy=g_twopoint_function_list[i2pt].pf2list[0][1];
        int pfpz=g_twopoint_function_list[i2pt].pf2list[0][2];




        struct stat fileStat;
        if(stat( filename, &fileStat) < 0 ) {
        /* Open an existing file. */
          fprintf ( stdout, "# [test_hdf5] create new file %s\n",filename );
          file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        } else {
          fprintf ( stdout, "# [test_hdf5] open existing file %s\n", filename );
          file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        }

        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/",  Ptot[0], Ptot[1], Ptot[2] );
        }
        status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

        status = H5Gget_objinfo (file_id, tagname, 0, NULL);
        if (status != 0){

           /* Create a group named "/MyGroup" in the file. */
           group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

           /* Close the group. */
           status = H5Gclose(group_id);

        }

        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu);
        }
        else{
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/",  Ptot[0],Ptot[1],Ptot[2],imu);
        }
        status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

        status = H5Gget_objinfo (file_id, tagname, 0, NULL);
        if (status != 0){

           /* Create a group named "/MyGroup" in the file. */
           group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

           /* Close the group. */
           status = H5Gclose(group_id);

        }
        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d",  Ptot[0],Ptot[1],Ptot[2],imu,ibeta);
        }
        status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

        status = H5Gget_objinfo (file_id, tagname, 0, NULL);
        if (status != 0){

           /* Create a group named "/MyGroup" in the file. */
           group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
           /* Close the group. */
           status = H5Gclose(group_id);

        }

        hsize_t dims[3];
        dims[0]=dimension_coeff;
        dims[1]=dimension_coeff;
        dims[2]=2;
        dataspace_id = H5Screate_simple(3, dims, NULL);
        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/a_data",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/a_data",  Ptot[0],
                                                       Ptot[1],Ptot[2],imu,ibeta);
        }
        /* Create a dataset in group "MyGroup". */
        dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


        double ***buffer_write=init_3level_dtable(dimension_coeff,dimension_coeff,2);

        for (int i=0; i < dimension_coeff; ++i){
          for (int j=0; j < dimension_coeff; ++j){
            buffer_write[i][j][0]=creal(projection_matrix_a[i][j]);
            buffer_write[i][j][1]=cimag(projection_matrix_a[i][j]);
          }
        }

        /* Write the first dataset. */
        status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

        /* Close the data space for the first dataset. */
        status = H5Sclose(dataspace_id);

        /* Close the first dataset. */
        status = H5Dclose(dataset_id);


        char * text_output=(char *)malloc(sizeof(char)*300);
        int **momentum_table= init_2level_itable( momentumlistsize, 6 );
        for (int ii=0; ii<momentumlistsize; ++ii){
          momentum_table[ii][0]=g_twopoint_function_list[i2pt].pf1list[ii][0];
          momentum_table[ii][1]=g_twopoint_function_list[i2pt].pf1list[ii][1];
          momentum_table[ii][2]=g_twopoint_function_list[i2pt].pf1list[ii][2];
          momentum_table[ii][3]=g_twopoint_function_list[i2pt].pf2list[ii][0];
          momentum_table[ii][4]=g_twopoint_function_list[i2pt].pf2list[ii][1];
          momentum_table[ii][5]=g_twopoint_function_list[i2pt].pf2list[ii][2];

        }
        int **spinf1table= init_2level_itable( g_twopoint_function_list[i2pt].number_of_gammas_f1, 2);
        for (int ii=0; ii<g_twopoint_function_list[i2pt].number_of_gammas_f1;++ii){
          spinf1table[ii][0]=g_twopoint_function_list[i2pt].list_of_gammas_f1[ii][0];
          spinf1table[ii][1]=g_twopoint_function_list[i2pt].list_of_gammas_f1[ii][1];

        }
        snprintf(text_output,300,"Pmatrix annihilation (imu=%d,ibeta=%d) Ptot=(%d,%d,%d)",imu,ibeta, Ptot[0], Ptot[1], Ptot[2]);
        rot_printf_matrix_non_zero_non_symmetric(projection_matrix_a, 1, dimension_coeff, momentum_table, momentumlistsize, spinf1table, g_twopoint_function_list[i2pt].number_of_gammas_f1, g_twopoint_function_list[i2pt].d , text_output, stdout);
        free(text_output);

        fini_3level_dtable(&buffer_write);

        /* We save the number of replicas */
        /* and the number of different momentum combinations */
        int N=dimension_coeff;
        double _Complex **projection_coeff_ORT= apply_gramschmidt ( projection_matrix_a, ANNIHILATION,  &N);

        hsize_t dims_linear;
        dims_linear=2;
        dataspace_id = H5Screate_simple(1, &dims_linear, NULL);
        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/Nreplicas_Nps_Ndimirrep",  Ptot[0],
                                                       Ptot[1],Ptot[2],imu,ibeta);
        }

        dataset_id = H5Dcreate2(file_id, tagname, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        int *buffer_linear=init_1level_itable(3);

        buffer_linear[0]=N;
        buffer_linear[1]=momentumlistsize;
        buffer_linear[2]=irrep_dim;

        /* Write the first dataset. */
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_linear[0]));

        /* Close the data space for the first dataset. */
        status = H5Sclose(dataspace_id);

        /* Close the first dataset. */
        status = H5Dclose(dataset_id);
 
        fini_1level_itable(&buffer_linear);

        hsize_t dims_quadratic[2];
        dims_quadratic[0]=momentumlistsize;
        dims_quadratic[1]=6;

        dataspace_id = H5Screate_simple(2, dims_quadratic, NULL);
        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/momlist_f1f2",  Ptot[0],
                                                       Ptot[1],Ptot[2],imu,ibeta);
        }

        /* Create a dataset in group "MyGroup". */
        dataset_id = H5Dcreate2(file_id, tagname, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Write the first dataset. */
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(momentum_table[0][0]));

        /* Close the data space for the first dataset. */
        status = H5Sclose(dataspace_id);

        /* Close the first dataset. */
        status = H5Dclose(dataset_id);

       
        if (N>0){
          text_output=(char *)malloc(sizeof(char)*300);
          snprintf(text_output,300,"Pmatrix annihilation orthonormalized (imu=%d,ibeta=%d)",imu,ibeta);
          rot_printf_matrix_non_zero_non_symmetric(projection_coeff_ORT, 1, N, momentum_table, momentumlistsize, spinf1table, g_twopoint_function_list[i2pt].number_of_gammas_f1, g_twopoint_function_list[i2pt].d , text_output, stdout);

          free(text_output);

          buffer_write=init_3level_dtable(N,dimension_coeff,2);

          for (int i=0; i < N; ++i){
            for (int j=0; j < dimension_coeff; ++j){
              buffer_write[i][j][0]=creal(projection_coeff_ORT[i][j]);
              buffer_write[i][j][1]=cimag(projection_coeff_ORT[i][j]);
            }
          }

          dims[0]=N;
          dims[1]=dimension_coeff;
          dims[2]=2;
          dataspace_id = H5Screate_simple(3, dims, NULL);
          if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
            snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/a_data_ort",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
          }
          else {
            snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/a_data_ort",  Ptot[0], Ptot[1], Ptot[2], imu, ibeta);
          }
          /* Create a dataset in group "MyGroup". */
          dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

          /* Write the first dataset. */
          status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

          /* Close the data space for the first dataset. */
          status = H5Sclose(dataspace_id);

          /* Close the first dataset. */
          status = H5Dclose(dataset_id);

          fini_3level_dtable(&buffer_write);
        
        
          fini_2level_ztable(&projection_coeff_ORT) ;
       
        }
        dataspace_id = H5Screate_simple(3, dims, NULL);
        if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
          snprintf ( tagname, 400,  "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/c_data",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
        }
        else {
          snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/c_data",  Ptot[0],Ptot[1],Ptot[2],imu,ibeta);
        }
        /* Create a dataset in group "MyGroup". */
        dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


        rot_mat_adj ( projection_matrix_c , projection_matrix_a , dimension_coeff  );

        text_output=(char *)malloc(sizeof(char)*300);
        snprintf(text_output,300,"Pmatrix creation (imu=%d,ibeta=%d)",imu,ibeta);
        rot_printf_matrix_non_zero_non_symmetric(projection_matrix_c, 0, dimension_coeff, momentum_table, momentumlistsize, spinf1table, g_twopoint_function_list[i2pt].number_of_gammas_f1, g_twopoint_function_list[i2pt].d , text_output, stdout);

        free(text_output);


        buffer_write=init_3level_dtable(dimension_coeff,dimension_coeff,2);


        for (int i=0; i < dimension_coeff; ++i){
          for (int j=0; j < dimension_coeff; ++j){
            buffer_write[i][j][0]=creal(projection_matrix_c[i][j]);
            buffer_write[i][j][1]=cimag(projection_matrix_c[i][j]);
          }
        }

        /* Write the first dataset. */
        status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));
        

        /* Close the data space for the first dataset. */
        status = H5Sclose(dataspace_id);

        /* Close the first dataset. */
        status = H5Dclose(dataset_id);

        fini_3level_dtable(&buffer_write);

        N=dimension_coeff;
        projection_coeff_ORT= apply_gramschmidt ( projection_matrix_c,CREATION,  &N);

        if (N>0){

          text_output=(char *)malloc(sizeof(char)*300);
          snprintf(text_output,300,"Pmatrix creation orthonormalized (imu=%d,ibeta=%d)",imu,ibeta);
          rot_printf_matrix_non_zero_non_symmetric(projection_coeff_ORT, 0, N, momentum_table, momentumlistsize, spinf1table, g_twopoint_function_list[i2pt].number_of_gammas_f1, g_twopoint_function_list[i2pt].d , text_output, stdout);

          free(text_output);


          buffer_write=init_3level_dtable(dimension_coeff,N,2);

          for (int i=0; i < dimension_coeff; ++i){
            for (int j=0; j < N; ++j){
              buffer_write[i][j][0]=creal(projection_coeff_ORT[i][j]);
              buffer_write[i][j][1]=cimag(projection_coeff_ORT[i][j]);
            }
          }

          dims[0]=dimension_coeff;
          dims[1]=N;
          dims[2]=2;
          dataspace_id = H5Screate_simple(3, dims, NULL);
          if (strcmp( g_twopoint_function_list[i2pt].name, "piN")==0){
             snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d_pi%dN%d/mu_%d/beta_%d/c_data_ort",  Ptot[0],Ptot[1],Ptot[2],pfpx*pfpx+pfpy*pfpy+pfpz*pfpz,pfnx*pfnx+pfny*pfny+pfnz*pfnz, imu, ibeta);
          }else {

             snprintf ( tagname, 400, "/pfx%dpfy%dpfz%d/mu_%d/beta_%d/c_data_ort",  Ptot[0],Ptot[1],Ptot[2],imu,ibeta);

          }

          /* Create a dataset in group "MyGroup". */
          dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

          /* Write the first dataset. */
          status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

          /* Close the data space for the first dataset. */
          status = H5Sclose(dataspace_id);

          /* Close the first dataset. */
          status = H5Dclose(dataset_id);

          fini_3level_dtable(&buffer_write);
        
        
          fini_2level_ztable(&projection_coeff_ORT) ;

        }
        fini_2level_ztable( &projection_matrix_a );
        fini_2level_ztable( &projection_matrix_c );
        fini_2level_itable( &momentum_table );
        fini_2level_itable( &spinf1table );


      }  // end of loop on imu

    }  // end of loop on ref_sink

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
