/****************************************************
 * p2gg_analyse_neutral
 *
 * NOTE:
 * 
 * this produces the SUM of up-type and down-type contribution
 *
 * parity-flavor averaged
 *
 * 8 pieces are added up
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <sys/time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#define MAIN_PROGRAM

#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "contract_cvc_tensor.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "uwerr.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const pgg_operator_type_tag[3][12]     = { "p-cvc-cvc" , "p-lvc-lvc" , "p-cvc-lvc" };

  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };

  /*                            gtg5 gxg5 gyg5 gzg5 */
  int const gamma_a_list[4] = {    6,   7,   8,   9 };

  /* vector, axial vector */
  int const gamma_va_list[8] = { 0,  1,  2,  3,  6,   7,   8,   9 };

  /*                             id  g5 */
  int const gamma_sp_list[2] = { 4  , 5 };



  /***********************************************************
   * sign for g5 Gamma^\dagger g5
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_g5d[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  /***********************************************************
   * sign for gt Gamma gt
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_t[16] ={  1, -1, -1, -1, +1, -1,  -1,  +1,  +1,  +1,  -1,  -1,  -1,  +1,  +1,  +1 };


  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  struct timeval ta, tb;
  int operator_type = -1;
  int write_data = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Wh?f:N:S:O:E:w:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_neutral] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_neutral] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [p2gg_analyse_neutral] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_neutral] operator_type set to %d\n", operator_type );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_analyse_neutral] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_neutral] write_data set to %d\n", write_data );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg_analyse_neutral] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [p2gg_analyse_neutral] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse_neutral] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse_neutral] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse_neutral] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse_neutral] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_analyse_neutral] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse_neutral] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse_neutral] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse_neutral] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse_neutral] comment %s\n", line );
      continue;
    }

    sscanf( line, "%c %d %d %d %d %d", 
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+5 );

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 1 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %c %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] );

      }
    }
  }

  /**********************************************************
   * sink momentum list modulo parity
   **********************************************************/
  int ** sink_momentum_list = init_2level_itable ( g_sink_momentum_number, 3 );
  memcpy ( sink_momentum_list[0], g_sink_momentum_list[0], 3 * sizeof(int) );
  int sink_momentum_number = 1;


  for ( int i = 1; i < g_sink_momentum_number; i++ ) {
    int have_pmom = 0;
    for ( int k = 0; k < sink_momentum_number; k++ ) {
      if ( ( sink_momentum_list[k][0] == -g_sink_momentum_list[i][0] ) &&
           ( sink_momentum_list[k][1] == -g_sink_momentum_list[i][1] ) &&
           ( sink_momentum_list[k][2] == -g_sink_momentum_list[i][2] ) ) {
        have_pmom = 1;
        break;
      }
    }
    if ( !have_pmom ) {
      memcpy ( sink_momentum_list[sink_momentum_number], g_sink_momentum_list[i], 3 * sizeof(int) );
      sink_momentum_number++;
    }
  }
  if ( g_verbose > 2 ) {
    for ( int i = 0; i < sink_momentum_number; i++ ) {
      fprintf( stdout, "# [p2gg_analyse_neutral] sink momentum %3d    %3d %3d %3d\n", i, 
          sink_momentum_list[i][0], sink_momentum_list[i][1], sink_momentum_list[i][2] );
    }
  }
  

  /**********************************************************
   **********************************************************
   **
   ** P -> gg 3-point function
   **
   **********************************************************
   **********************************************************/

  for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) 
  {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_momentum][0],
      g_seq_source_momentum_list[iseq_source_momentum][1],
      g_seq_source_momentum_list[iseq_source_momentum][2] };

      int const sequential_source_gamma_id = 4;

    for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
    {

      int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

      double ****** pgg = init_6level_dtable ( num_conf, num_src_per_conf, sink_momentum_number, 4, 4, 2 * T );
      if ( pgg == NULL ) {
        fprintf(stderr, "[p2gg_analyse_neutral] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      for ( int iflavor = 0; iflavor <= 1; iflavor++ )
      {


          /**********************************************************
           * loop on momenta
           **********************************************************/
          for ( int isink_momentum = 0; isink_momentum < sink_momentum_number; isink_momentum++ ) {

            int const sink_momentum[3] = {
                sink_momentum_list[isink_momentum][0],
                sink_momentum_list[isink_momentum][1],
                sink_momentum_list[isink_momentum][2] };

            for ( int ipsign = 0; ipsign <= 1; ipsign++ )
            {

              int const psign = 1 - 2 * ipsign;

              sprintf ( filename, "%s/%s.%s.fl%d.qx%d_qy%d_qz%d.gseq_%d.tseq_%d.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2,
                  pgg_operator_type_tag[operator_type], iflavor, 
                  psign * seq_source_momentum[0], psign * seq_source_momentum[1], psign * seq_source_momentum[2],
                  sequential_source_gamma_id,
                  sequential_source_timeslice,
                  psign * sink_momentum[0], psign * sink_momentum[1], psign * sink_momentum[2] );

              fprintf ( stdout, "# [p2gg_analyse_neutral] filename = %s %s %d\n", filename, __FILE__, __LINE__ );

              double *** buffer = init_3level_dtable( 4, 4, 2 * T );
              if( buffer == NULL ) {
                fprintf(stderr, "[p2gg_analyse_neutral] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(15);
              }
      
              /***********************************************************
               * loop on configs and source locations per config
               ***********************************************************/
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
      
                  /***********************************************************
                   * copy source coordinates
                   ***********************************************************/
                  int const gsx[4] = {
                      conf_src_list[iconf][isrc][2],
                      conf_src_list[iconf][isrc][3],
                      conf_src_list[iconf][isrc][4],
                      conf_src_list[iconf][isrc][5] };
      
                  /***********************************************
                   * reader for aff input file
                   ***********************************************/
                  gettimeofday ( &ta, (struct timezone *)NULL );

                  if ( operator_type == 1 ) {

                    gettimeofday ( &ta, (struct timezone *)NULL );

                    char key[400];

                    sprintf ( key , "/stream_%c/conf_%d/t%d_x%d_y%d_z%d", 
                        conf_src_list[iconf][isrc][0],
                        conf_src_list[iconf][isrc][1],
                        conf_src_list[iconf][isrc][2],
                        conf_src_list[iconf][isrc][3],
                        conf_src_list[iconf][isrc][4],
                        conf_src_list[iconf][isrc][5] );

                    if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_neutral] key = %s\n", key );

                    exitstatus = read_from_h5_file ( (void*)buffer[0][0], filename, key, "double", io_proc );
                    if( exitstatus != 0 ) {
                      fprintf(stderr, "[p2gg_analyse_neutral] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                          filename, key,
                          exitstatus, __FILE__, __LINE__);
                      EXIT(105);
                    }

                    gettimeofday ( &tb, (struct timezone *)NULL );
                    show_time ( &ta, &tb, "p2gg_analyse_neutral", "read-pgg-tensor-components-h5", g_cart_id == 0 );

                    /**********************************************************
                     * loop on shifts in directions mu, nu
                     **********************************************************/
                    for( int mu = 0; mu < 4; mu++) {
                      for( int nu = 0; nu < 4; nu++) {
            
                        /**********************************************************
                         * signs for g5-hermiticity and parity for current and source
                         **********************************************************/
                        int s5d_sign[2] = { 0, 0 };
                        int st_sign[2]  = { 0, 0 };
     
                        s5d_sign[0] = 
                               sigma_g5d[ sequential_source_gamma_id ]
                             * sigma_g5d[ gamma_v_list[mu] ]
                             * sigma_g5d[ gamma_v_list[nu] ];
                        s5d_sign[1] = s5d_sign[0];
      
                        st_sign[0] = 
                               sigma_t[ sequential_source_gamma_id ]
                             * sigma_t[ gamma_v_list[mu] ]
                             * sigma_t[ gamma_v_list[nu] ];
                        st_sign[1] = st_sign[0];
      
                        if ( g_verbose > 5 ) fprintf (stdout, "# [p2gg_analyse_neutral] mu %d nu %d   s5d %2d  %2d    st %2d  %2d\n", mu, nu, 
                            s5d_sign[0], s5d_sign[1], st_sign[0], st_sign[1] );

                        double const fl_ps_sign[2][2][2] =  {
                          { { -1, -1}, { +s5d_sign[0], -s5d_sign[0] } }, 
                          { { +1, +1}, { -s5d_sign[0], +s5d_sign[0] } } };


      
                    /**********************************************************
                     * current vertex momentum 4-vector; 0th component stays zero
                     **********************************************************/
                      double const p[4] = { 0., 
                          TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
                          TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
                          TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };
      
                      if (g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse_neutral] p = %25.16e %25.16e %25.16e %25.16e\n", p[0], p[1], p[2], p[3] ); 
      
                    /**********************************************************
                     * final vertex momentum 4-vector; 0th component stays zero
                     **********************************************************/
                      double const q[4] = { 0., 
                          TWO_MPI * (double)seq_source_momentum[0] / (double)LX_global,
                          TWO_MPI * (double)seq_source_momentum[1] / (double)LY_global,
                          TWO_MPI * (double)seq_source_momentum[2] / (double)LZ_global };
                      
                      if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse_neutral] q = %25.16e %25.16e %25.16e %25.16e\n", q[0], q[1], q[2], q[3] ); 
            
                      double p_phase = 0., q_phase = 0.;
      
                      p_phase = -( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] );
                      q_phase = -( q[0] * gsx[0] + q[1] * gsx[1] + q[2] * gsx[2] + q[3] * gsx[3] );
      
                      double _Complex const p_ephase  = cexp ( p_phase * I );
                      double _Complex const pm_ephase = conj ( p_ephase ); 
      
                      double _Complex const q_ephase  = cexp ( q_phase * I );
                      double _Complex const qm_ephase = conj ( q_ephase );
            
                      if ( g_verbose > 4 ) {
                        fprintf ( stdout, "# [p2gg_analyse_neutral] p %3d %3d %3d x %3d %3d %3d %3d p_phase %25.16e   p_ephase %25.16e %25.16e     mp_ephase %25.16e %25.16e\n",
                            sink_momentum[0], sink_momentum[1], sink_momentum[2], 
                            gsx[0], gsx[1], gsx[2], gsx[3], p_phase, creal( p_ephase ), cimag( p_ephase ), creal( pm_ephase ), cimag( pm_ephase ) );
                        
                        fprintf ( stdout, "# [p2gg_analyse_neutral] q %3d %3d %3d x %3d %3d %3d %3d q_phase %25.16e   q_ephase %25.16e %25.16e     qm_ephase %25.16e %25.16e\n",
                            sink_momentum[0], sink_momentum[1], sink_momentum[2], 
                            gsx[0], gsx[1], gsx[2], gsx[3], q_phase, creal( q_ephase ), cimag( q_ephase ), creal( qm_ephase ), cimag( qm_ephase ) );
                      }
            
                      /**********************************************************
                       * sort data from buffer into pgg,
                       * multiply source phase
                       **********************************************************/

#pragma omp parallel for
                      for ( int it = 0; it < T; it++ ) {
      
                        /**********************************************************
                         * order from source time
                         **********************************************************/
                        int const tt = ( it - gsx[0] + T_global ) % T_global; 
            
                        /**********************************************************
                         * add the two flavor components
                         **********************************************************/
                        double _Complex ztmp = ( 
                              fl_ps_sign[iflavor][ipsign][0] * buffer[mu][nu][2*it  ] 
                            + fl_ps_sign[iflavor][ipsign][1] * buffer[mu][nu][2*it+1] * I 
                            ) * p_ephase;
      
                          /**********************************************************
                           * multiply source phase from pion momentum
                           **********************************************************/
                          ztmp *= q_ephase;
      
                          /**********************************************************
                           * add up original and Parity-flavor transformed
                           **********************************************************/
                          ztmp += st_sign[0] * conj ( ztmp ); 

                          /**********************************************************
                           * AVERAGE over parity-flavor, DO NOT ADD,
                           * so we add a factor of 1/2
                           **********************************************************/
                          ztmp *= 0.5;
      
                          /**********************************************************
                           * write into pgg
                           **********************************************************/
                          pgg[iconf][isrc][isink_momentum][mu][nu][2*tt  ] += creal( ztmp );
                          pgg[iconf][isrc][isink_momentum][mu][nu][2*tt+1] += cimag( ztmp );
                        }  /* end of loop on timeslices */
      
                    }  /* end of loop on direction nu */
                    }  /* end of loop on direction mu */

                  }  /* end of operator_type */

                }  /* end of loop on src */

              }  /* end of loop on conf */

              fini_3level_dtable( &buffer );

            }  /* end of loop on mom sign  */
          }  /* end of loop on sink momenta */

      }  /* end of loop on flavor */


      /**********************************************************
       * show all data
       **********************************************************/
      if ( g_verbose > 5 ) {
        gettimeofday ( &ta, (struct timezone *)NULL );

        for ( int iconf = 0; iconf < num_conf; iconf++ )
        {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
          {
            for ( int imom = 0; imom < sink_momentum_number; imom++ ) {

              fprintf ( stdout , "# /%s/stream_%c/c%d/t%dx%dy%dz%d/qx%dqy%dqz%d/gseq%d/tseq%d/px%dpy%dpz%d\n", pgg_operator_type_tag[operator_type],
                  conf_src_list[iconf][isrc][0],
                  conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2],
                  conf_src_list[iconf][isrc][3],
                  conf_src_list[iconf][isrc][4],
                  conf_src_list[iconf][isrc][5],
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  sequential_source_gamma_id, sequential_source_timeslice,
                  sink_momentum_list[imom][0], sink_momentum_list[imom][1], sink_momentum_list[imom][2] );

              for ( int mu = 0; mu < 4; mu++ ) {
              for ( int nu = 0; nu < 4; nu++ ) {
                for ( int it = 0; it < T; it++ ) {
                  fprintf ( stdout, "c %d %d    %3d    %25.16e %25.16e\n", 
                      mu, nu, it, pgg[iconf][isrc][imom][mu][nu][2*it], pgg[iconf][isrc][imom][mu][nu][2*it+1] );
                }
              }}
            }
          }
        }
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "p2gg_analyse_neutral", "show-all-data", g_cart_id == 0 );
      }

      /**********************************************************
       * source average
       **********************************************************/
      double ***** pgg_src_avg = init_5level_dtable ( num_conf, sink_momentum_number, 4, 4, 2 * T );
      if ( pgg_src_avg == NULL ) {
        fprintf(stderr, "[p2gg_analyse_neutral] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int i = 0; i < sink_momentum_number * 32 * T_global; i++ ) {
          pgg_src_avg[iconf][0][0][0][i] = 0.;
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            pgg_src_avg[iconf][0][0][0][i] += pgg[iconf][isrc][0][0][0][i];
          }
          pgg_src_avg[iconf][0][0][0][i] /= (double)num_src_per_conf;
        }
      }


      /****************************************
       * statistical analysis for orbit average
       *
       * ASSUMES MOMENTUM LIST IS AN ORBIT AND
       * SEQUENTIAL MOMENTUM IS ZERO
       ****************************************/
      for ( int ireim = 0; ireim < 2; ireim++ ) {

          double ** data = init_2level_dtable ( num_conf, T_global );
 
          int const dim[2] = { num_conf, T_global };
          antisymmetric_orbit_average_spatial ( data, pgg_src_avg, dim, sink_momentum_number, sink_momentum_list, ireim );

          char obs_name[200];
          sprintf ( obs_name, "pgg_conn.%s.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", pgg_operator_type_tag[operator_type],
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                sink_momentum_list[0][0], sink_momentum_list[0][1], sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          if ( num_conf >= 6 ) {
            exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[p2gg_analyse_neutral] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          }

          if ( write_data == 1 ) {
            sprintf ( filename, "%s.corr", obs_name );

            FILE * ofs = fopen ( filename, "w" );
            if ( ofs == NULL ) {
              fprintf ( stdout, "[p2gg_analyse_neutral] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
              EXIT(12);
            }

            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int it = 0; it < T_global; it++ ) 
              /* for ( int tau = -T_global/2+1; tau <= T_global/2; tau++ )  */
              {
                /* int const it = ( tau < 0 ) ? tau + T_global : tau; */

                fprintf ( ofs, "%5d %25.16e %c %8d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
              }
            }
            fclose ( ofs );

          }  /* end of if write data */

          fini_2level_dtable ( &data );

      }  /* end of loop on real / imag */

      /**********************************************************
       * free p2gg table
       **********************************************************/
      fini_6level_dtable ( &pgg );
      fini_5level_dtable ( &pgg_src_avg );

    }  /* end of loop on sequential source timeslices */

  }  /* end of loop on seq source momentum */


  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );
  fini_2level_itable ( &sink_momentum_list );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_analyse_neutral] %s# [p2gg_analyse_neutral] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse_neutral] %s# [p2gg_analyse_neutral] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
