/****************************************************
 * piq_analyse 
 ****************************************************/

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
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif


#define _TWOP_CYD_H5    1
#define _TWOP_CYD_H5_TR 0

#define _TWOP_STATS  1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}


/**********************************************************
 *
 **********************************************************/
inline void write_data_real ( double ** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1 ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real] Error from fopen %s %d\n",  __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
    fprintf ( ofs, "# %c %6d\n", lst[i0][0][0], lst[i0][0][1] );
    for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1] );
    }
  }

  fclose ( ofs );
}  /* end of write_data_real */


/**********************************************************
 *
 **********************************************************/
inline void write_data_real2_reim ( double **** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real2_reim] Error from fopen %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
  for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
    fprintf ( ofs , "# %c %6d %3d %3d %3d %3d\n", lst[i0][i1][0], lst[i0][i1][1], lst[i0][i1][2], lst[i0][i1][3], lst[i0][i1][4], lst[i0][i1][5] );

    for ( unsigned int i2 = 0; i2 < n2; i2++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1][i2][ri] );
    }
  }}
  fclose ( ofs );
}  /* end of write_data_real2_reim */

/**********************************************************
 *
 **********************************************************/
inline void src_avg_real2_reim ( double ** data, double ****corr, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

#pragma omp parallel for
  for ( unsigned int iconf = 0; iconf < n0; iconf++ ) {
    for ( unsigned int it = 0; it < n2; it++ ) {
      double dtmp = 0.;

      for ( unsigned int isrc = 0; isrc < n1; isrc++ ) {
        dtmp += corr[iconf][isrc][it][ri];
      }
      data[iconf][it] = dtmp / (double)n1;
    }
  }
}  /* end of src_avg_real2_reim */

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  /* int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 }; */

  char const reim_str[2][3] = { "re", "im" };

  char const flavor_tag[3][3] = { "uu", "dd" , "ud" };
  char const flavor_output_tag[3][3] = { "ud", "du" , "uu" };

  double const TWO_MPI = 2. * M_PI;

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;
  int write_data = 0;

  struct timeval ta, tb, start_time, end_time;

  int flavor_id = -1;

  int muval_set = 0;
  char muval_tag[12] = "NA";

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:E:w:i:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [piq_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [piq_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [piq_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [piq_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [piq_analyse] write_date set to %d\n", write_data );
      break;
    case 'i':
      flavor_id = atoi ( optarg );
      fprintf ( stdout, "# [piq_analyse] flavor_id set to %d\n", flavor_id );
      break;
    case 'm':
      muval_set = 1;
      strcpy ( muval_tag, optarg );
      fprintf ( stdout, "# [piq_analyse] muval_tag set (%d) to %s\n", muval_set, muval_tag );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [piq_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [piq_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [piq_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [piq_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[piq_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[piq_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[piq_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [piq_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[piq_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[piq_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [piq_analyse] comment %s\n", line );
      continue;
    }
    int itmp[5];
    char ctmp;

    sscanf( line, "%c %d %d %d %d %d", &ctmp, itmp, itmp+1, itmp+2, itmp+3, itmp+4 );

    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][0] = (int)ctmp;
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][1] = itmp[0];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][2] = itmp[1];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][3] = itmp[2];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][4] = itmp[3];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][5] = itmp[4];

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 3 ) {
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
   * gamma matrices
   **********************************************************/
  init_gamma_matrix ();
  /*                                        0         1         2       3       4       5       6       7      8     9 */
  int const gamma_num = 10;
  char const gamma_tag[10][6]    = {     "gx",     "gy",     "gz",    "gt", "gxg5", "gyg5", "gzg5", "gtg5", "g5", "id" };
  int const gamma_id[10]         = {       14,       13,       11,       7,      1,      2,      4,      8,    0,   15 };

  int const gamma_binary_id[10]  = {        1,        2,        4,       8,     14,     13,     11,      7,   15,    0 };

  /*                               5x = yzt  5y = xzt  z5 = xyt 5t = xyz       x       y       z       t    id     5 */
  int const gamma_sign_snk[10] = {       -1,       +1,       -1,      +1,     -1,     -1,     -1,     -1,   +1,   +1 };

  /*                               x5 = yzt  y5 = xzt  z5 = xyt t5 = xyz       x       y       z       t    id     5 */
  int const gamma_sign_src[10] = {       +1,       -1,       +1,      -1,     +1,     +1,     +1,     +1,   +1,   +1 };

  /* int gamma_src_snk_num[] */
  /* int gamma_src_snk_pairs[18][2] = {
    { 0, 0},
    { 1, 1},
    { 2, 2},
    { 3, 3},
    { 4, 4},
    { 5, 5},
    { 6, 6},
    { 7, 7},
    { 8, 8},
    { 9, 9},
    { 3, 8},
    { 8, 3},
    { 3, 9},
    { 9, 3},
    { 7, 8},
    { 8, 7},
    { 7, 9},
    { 9, 7} }; */


  gamma_matrix_type gamma_mat[10];

  for ( int i = 0; i < gamma_num; i++ ) 
  {
    gamma_matrix_ukqcd_binary ( &(gamma_mat[i]), gamma_id[i] );

    if ( g_verbose > 1 ) {
      gamma_matrix_printf ( &(gamma_mat[i]), gamma_tag[i], stdout );
    }
  }


  /**********************************************************
   **********************************************************
   ** 
   ** READ DATA
   ** 
   **********************************************************
   **********************************************************/

  /***********************************************************
   * read twop function data
   ***********************************************************/

  /**********************************************************
   * VV 2-pt projected
   **********************************************************/

  double _Complex *** twop_proj = init_3level_ztable ( num_conf, T_global, gamma_num );
  if( twop_proj == NULL ) {
    fprintf ( stderr, "[piq_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  /***********************************************************/
  /***********************************************************/

#if _TWOP_CYD_H5
  double _Complex ***** twop = init_5level_ztable ( T, 4, 4, 4, 4 );
  if( twop == NULL ) {
    fprintf ( stderr, "[piq_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  double _Complex * twop_gamma = init_1level_ztable ( gamma_num );
  if( twop_gamma == NULL ) {
    fprintf ( stderr, "[piq_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    if ( muval_set )  {
      sprintf( filename, "%s/%.4d_r%c_%s.h5", filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] , muval_tag );
    } else {
      sprintf( filename, "%s/%.4d_r%c.h5", filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
    }
    if ( g_verbose > 4 ) fprintf ( stdout, "# [piq_analyse] filename = %s  %s %d\n", filename, __FILE__, __LINE__ );
    
    char h5_tag[200];

    double ******* buffer = init_7level_dtable ( T, 1, 4, 4, 4, 4, 2 );
    if ( buffer == NULL ) {
      fprintf ( stderr, "[] Error from init_Xlevel_dtable   %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
 
      if ( muval_set )  {
        sprintf ( h5_tag, "/%s_open/%s_id%.2d_st%.3d", flavor_tag[flavor_id], muval_tag, conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][2] );
      } else {
        sprintf ( h5_tag, "/%s_open/id%.2d_st%.3d", flavor_tag[flavor_id], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][2] );
      }

      if ( g_verbose > 4 ) fprintf ( stdout, "# [piq_analyse] h5 tag = %s %s %d\n", h5_tag, __FILE__, __LINE__  );

      exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0][0][0][0]), filename, h5_tag, "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[piq_analyse] Error from read_from_h5_file for %s %s, status was %d %s %d\n", filename, h5_tag, exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
#pragma omp parallel for
      for ( int it = 0; it < T; it++ ) 
      {
        int const offset = it *  4*4*4*4;

        for ( int ix = 0; ix < 4*4*4*4; ix++ ) 
        {
          /* ordered from source ? */
          // twop[it][0][0][0][ix] += buffer[it][0][0][0][0][0][2*ix] + buffer[it][0][0][0][0][0][2*ix+1] * I;

          twop[0][0][0][0][offset + ix] += buffer[0][0][0][0][0][0][2*(offset+ix)] + buffer[0][0][0][0][0][0][2*(offset+ix)+1] * I;
        }  /* end of loop on ix */

        /* TEST */
        if( it == 0 ) 
        {
          for ( int imu = 0; imu < 1 ; imu++ )
          {
            double _Complex ztmp = 0;
            for ( int i1 = 0; i1 < 4; i1++ ) {
            for ( int i2 = 0; i2 < 4; i2++ ) {
            for ( int k1 = 0; k1 < 4; k1++ ) {
            for ( int k2 = 0; k2 < 4; k2++ ) {
              // ztmp += ( buffer[it][0][k1][i1][i2][k2][0] + buffer[it][0][k1][i1][i2][k2][1] * I ) * gamma_mat[imu].m[i1][i2] * gamma_mat[imu].m[k1][k2];
              int const ix =   ( ( k1 * 4 + i1 ) * 4 + i2 ) * 4 + k2;
              // ztmp += ( buffer[it][0][0][0][0][0][2*ix] + buffer[it][0][0][0][0][0][2*ix+1] * I ) * gamma_mat[imu].m[i1][i2] * gamma_mat[imu].m[k1][k2];

              double _Complex const w = buffer[0][0][0][0][0][0][2*(offset+ix)] + buffer[0][0][0][0][0][0][2*(offset+ix)+1] * I ;

              // ztmp += ( buffer[0][0][0][0][0][0][2*(offset+ix)] + buffer[0][0][0][0][0][0][2*(offset+ix)+1] * I ) * gamma_mat[imu].m[i1][i2] * gamma_mat[imu].m[k1][k2];
              ztmp += w * gamma_mat[imu].m[i1][i2] * gamma_mat[imu].m[k1][k2];
            }}}}

            twop_gamma[imu] = ztmp * gamma_sign_snk[imu] * gamma_sign_src[imu];

            fprintf ( stdout, "%6s %c %6d %6d %25.16e %25.16e\n", gamma_tag[imu],
                conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], isrc,
                creal ( twop_gamma[imu] ), cimag (  twop_gamma[imu] ) );
          }
        }
        /* END TEST */
        

      }  /* end of loop on timeslices */

    }  /* end of loop on sources  */

    for ( int ix = 0; ix < T*4*4*4*4; ix++ )
    {
      twop[0][0][0][0][ix] /= (double)num_src_per_conf;
    }

    fini_7level_dtable ( &buffer );

#pragma omp parallel for 
    for ( int it = 0; it < T; it++ ) 
    {

      // for ( int imu = 0; imu < gamma_num; imu++ ) 
      for ( int imu = 0; imu < 1; imu++ ) 
      {
        double _Complex ztmp = 0;
        for ( int i1 = 0; i1 < 4; i1++ ) {
        for ( int i2 = 0; i2 < 4; i2++ ) {
        for ( int k1 = 0; k1 < 4; k1++ ) {
        for ( int k2 = 0; k2 < 4; k2++ ) {
          ztmp += twop[it][k1][i1][i2][k2] * gamma_mat[imu].m[i1][i2] * gamma_mat[imu].m[k1][k2];
        }}}}
       
        // twop_proj[iconf][it][imu] = ztmp * gamma_sign[imu];
        twop_proj[iconf][it][imu] = ztmp * gamma_sign_snk[imu] * gamma_sign_src[imu];

        if ( it == 0 ) {
          fprintf ( stdout, "a %6s %c %6d %25.16e %25.16e\n", gamma_tag[imu],
                  conf_src_list[iconf][0][0], conf_src_list[iconf][0][1],
                  creal ( twop_proj[iconf][it][imu] ), cimag (  twop_proj[iconf][it][imu] ) );
        }



      }  /* end of loop on gf-gi */
    }  /* end of loop on timeslices */
  }  /* end of loop on configurations */

  fini_5level_ztable ( &twop );
  fini_1level_ztable ( &twop_gamma );

#endif  /* end of _TWOP_CYD_H5 */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_CYD_H5_TR
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    if ( muval_set )  {
      sprintf( filename, "%s/%.4d_r%c_%s.h5", filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] , muval_tag );
    } else {
      sprintf( filename, "%s/%.4d_r%c.h5", filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
    }
    if ( g_verbose > 1 ) fprintf ( stdout, "# [piq_analyse] filename = %s  %s %d\n", filename, __FILE__, __LINE__ );
    
    char h5_tag[200];

    double **** buffer = init_4level_dtable ( T, 1, 16, 2 );
    if ( buffer == NULL ) {
      fprintf ( stderr, "[piq_analyse] Error from init_Xlevel_dtable   %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
 
      if ( muval_set )  {
        sprintf ( h5_tag, "/%s/%s_id%.2d_st%.3d", flavor_tag[flavor_id], muval_tag, conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][2] );
      } else {
        sprintf ( h5_tag, "/%s/id%.2d_st%.3d", flavor_tag[flavor_id], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][2] );
      }

      if ( g_verbose > 4 ) fprintf ( stdout, "# [piq_analyse] h5 tag = %s %s %d\n", h5_tag, __FILE__, __LINE__  );

      exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0]), filename, h5_tag, "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[piq_analyse] Error from read_from_h5_file for %s %s, status was %d %s %d\n", filename, h5_tag, exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
#pragma omp parallel for
      for ( int it = 0; it < T; it++ ) 
      {
        for ( int imu = 0; imu < gamma_num; imu++ )
        {
          int const mu = gamma_binary_id[imu];

          twop_proj[iconf][it][imu] += buffer[it][0][mu][0] + buffer[it][0][mu][1] * I;
        }
      }

    }  /* end of loop on sources  */

    for ( int ix = 0; ix < T * gamma_num; ix++ )
    {
      twop_proj[iconf][0][ix] /= (double)num_src_per_conf;
    }

    fini_4level_dtable ( &buffer );

  }  /* end of loop on configurations */

#endif  /* end of _TWOP_CYD_H5_TR */

  /***********************************************************/
  /***********************************************************/

  /**********************************************************
   * 
   * STATISTICAL ANALYSIS of twop
   *   orbit-averaged
   *   source-averaged
   * 
   **********************************************************/
  for ( int ig = 0; ig < gamma_num; ig++ )
  {
    for ( int ireim = 0; ireim < 2; ireim++ )
    {

      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf ( stderr, "[piq_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
        EXIT(1);
      }
        
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {

          int const itt = ( T_global - it ) % T_global;

          double const a[2] = { creal ( twop_proj[iconf][ it][ig] ), cimag ( twop_proj[iconf][ it][ig] )  };
          double const b[2] = { creal ( twop_proj[iconf][itt][ig] ), cimag ( twop_proj[iconf][itt][ig] )  };

          // data[iconf][it] = 0.5 * ( ((double*)&(twop_proj[iconf][it][ig]))[ireim] + ((double*)&(twop_proj[iconf][itt][ig]))[ireim] );
          data[iconf][it] = 0.5 * ( a[ireim] + b[ireim] );
        }
      }

      char obs_name[400];
      sprintf( obs_name, "twop.%s.%s-%s.%s", flavor_output_tag[flavor_id], gamma_tag[ig], gamma_tag[ig], reim_str[ireim] );

      /**********************************************************
       * write data to ascii file
       **********************************************************/
      if ( write_data == 1 ) {
        sprintf ( filename, "%s.corr", obs_name );

        FILE * fs = fopen ( filename, "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          }
        }
        fclose ( fs );
      }  /* end of if write data */

      if ( num_conf < 6 ) {
        fprintf ( stderr, "[piq_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      } else {

        /**********************************************************
         * apply UWerr analysis
         **********************************************************/
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piq_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        /**********************************************************
         * acosh ratio for m_eff
         **********************************************************/
        int const Thp1 = T_global / 2 + 1;
        for ( int itau = 1; itau < Thp1/4; itau++ ) {
          int narg = 3;
          int arg_first[3] = { 0, 2 * itau, itau };
          int arg_stride[3] = {1,1,1};
          int nT = Thp1 - 2 * itau;

          char obs_name2[500];
          sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );

          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piq_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
        }
      }

      fini_2level_dtable ( &data );
    }  /* end of loop on reim */

  }


  fini_3level_ztable ( &twop_proj );


  /**********************************************************
   * free and finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "piq_analyse", "runtime", g_cart_id == 0 );

  return(0);
}
