/****************************************************
 * xg_exdefl_analyse 
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

#define _LOOP_ANALYSIS

#define _XG_PION
#undef _XG_NUCLEON

#define _RAT_METHOD

#define _TWOP_STATS

#define MAX_SMEARING_LEVELS 40

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
  
  const double TWO_MPI = 2. * M_PI;

   char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz"
  };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };



  char const reim_str[2][3] = { "re", "im" };

  /* char const correlator_prefix[1][20] = { "local-local" }; */

  /* char const flavor_tag[2][20]        = { "d-gf-u-gi" , "u-gf-d-gi" }; */


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 1;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;
  int write_data = 0;
  int evecs_num = 0;
  struct timeval ta, tb;
  unsigned int stout_level_iter[MAX_SMEARING_LEVELS];
  double stout_level_rho[MAX_SMEARING_LEVELS];
  unsigned int stout_level_num = 0;
  int evecs_use_min = 0, evecs_use_max = -1, evecs_use_step = 0, evecs_use_nstep = 0;


 #ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:R:E:w:V:U:s:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] write_date set to %d\n", write_data );
      break;
    case 'V':
      evecs_num = atoi ( optarg );
      fprintf ( stdout, "# [xg_exdefl_analyse] evecs_num set to %d\n", evecs_num );
      break;
    case 'U':
      sscanf( optarg, "%d,%d,%d", &evecs_use_min, &evecs_use_step, &evecs_use_max );
      fprintf ( stdout, "# [xg_exdefl_analyse] evecs_use set to %d --- %d --- %d\n", evecs_use_min, evecs_use_step, evecs_use_max );
      break;
    case 's':
      sscanf ( optarg, "%d,%lf", stout_level_iter+stout_level_num, stout_level_rho+stout_level_num );
      fprintf ( stdout, "# [xg_exdefl_analyse] stout_level %d  iter %2d  rho %6.4f \n", stout_level_num, stout_level_iter[stout_level_num], stout_level_rho[stout_level_num] );
      stout_level_num++;
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
  /* fprintf(stdout, "# [xg_exdefl_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [xg_exdefl_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [xg_exdefl_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [xg_exdefl_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[xg_exdefl_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[xg_exdefl_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[xg_exdefl_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [xg_exdefl_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[xg_exdefl_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[xg_exdefl_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [xg_exdefl_analyse] comment %s\n", line );
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

 
  /***********************************************************
   * number of evecs accum steps
   ***********************************************************/
  evecs_use_nstep = ( evecs_use_max - evecs_use_min + evecs_use_step ) / evecs_use_step;
  fprintf ( stdout, "# [xg_exdefl_analyse] number of evecs accum stesp %d\n", evecs_use_nstep );


  /***********************************************************
   * loop on gamma at sink
   ***********************************************************/
  for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

  /***********************************************************
   * loop on gamma at source
   ***********************************************************/
  for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

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
    double ***** twop = NULL;

    twop = init_5level_dtable ( g_sink_momentum_number, num_conf, evecs_use_nstep, T_global, 2 * T_global  );
    if( twop == NULL ) {
      fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }

#ifdef _XG_PION
    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
  
      Nconf = conf_src_list[iconf][0][1];
 
      gettimeofday ( &ta, (struct timezone *)NULL );
  
      /***********************************************************
       * open AFF reader
       ***********************************************************/
      struct AffNode_s *affn = NULL, *affdir = NULL;
    
      sprintf( filename, "./stream_%c/%s/%d/%s.pref_%d_%d_%d.%.4d.nev%d.aff",
          conf_src_list[iconf][0][0], 
          filename_prefix,
          conf_src_list[iconf][0][1], 
          filename_prefix2, g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
          conf_src_list[iconf][0][1], 
          evecs_num );

      fprintf(stdout, "# [xg_exdefl_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
  
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }
  
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "xg_exdefl_analyse", "open-init-aff-reader", g_cart_id == 0 );
  
      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int inev = 0; inev < evecs_use_nstep; inev++ ) {

          int const evecs_use_num = evecs_use_min + inev * evecs_use_step;
  
          gettimeofday ( &ta, (struct timezone *)NULL );

          /***********************************************************
           * build key
           * /jj/g4_g4/px0_py0_pz0/qx0_qy0_qz0/nev50
           ***********************************************************/
  
          sprintf( key, "/jj/g%d_g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d",
              g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
               g_sink_momentum_list[ipf][0],  g_sink_momentum_list[ipf][1],  g_sink_momentum_list[ipf][2],
              -g_sink_momentum_list[ipf][0], -g_sink_momentum_list[ipf][1], -g_sink_momentum_list[ipf][2],
              evecs_use_num );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_exdefl_analyse] key = %s\n", key );
  
          affdir = aff_reader_chpath (affr, affn, key );
          if( affdir == NULL ) {
            fprintf(stderr, "[xg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
            EXIT(105);
          }
  
          uint32_t uitems = T_global * T_global;

          exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(twop[ipf][iconf][inev][0]), uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[xg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
  
          /***********************************************************
           * source phase is already added
           ***********************************************************/
  
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "xg_exdefl_analyse", "read-twop-aff", g_cart_id == 0 );
  
        }  /* end of loop on evecs use acuum steps */
      }  /* end of loop on sink momenta */
  
      /**********************************************************
       * close the reader
       **********************************************************/
      aff_reader_close ( affr );
  
    }  /* end of loop on configs */
#endif  /* end of _XG_PION */

    /**********************************************************
     *
     * average 2-pt over momentum orbit
     *
     **********************************************************/

    double **** twop_orbit = init_4level_dtable ( num_conf, evecs_use_nstep, T_global, 2 * T_global  );
    if( twop_orbit == NULL ) {
      fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }

    for ( int inev = 0; inev < evecs_use_nstep; inev++ ) {

      int const evecs_use_num = evecs_use_min + inev * evecs_use_step;

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        /* averaging starts here */
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int it1 = 0; it1 < T_global; it1++ ) {
          for ( int it2 = 0; it2 < T_global; it2++ ) {
            int const idt = ( it1 - it2 + T_global ) % T_global;
            twop_orbit[iconf][inev][it1][2*it2  ] += twop[imom][iconf][inev][it1][2*it2  ];
            twop_orbit[iconf][inev][it1][2*it2+1] += twop[imom][iconf][inev][it1][2*it2+1];
          }}  /* end of loop on it1,2 */
        }  /* end of loop on imom */
        for ( int it = 0; it < 2*T_global*T_global; it++ ) {
          twop_orbit[iconf][0][0][it] /= (double)( g_sink_momentum_number * T_global );
        }
      }  /* end of loop on iconf */

      /**********************************************************
       * write orbit-averaged data to ascii file, per source
       **********************************************************/

      char obs_name_prefix[200];
      sprintf ( obs_name_prefix, "twop.orbit.src-avg.gf_%s.gi_%s.px%d_py%d_pz%d.nev%d", 
              gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
              g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], evecs_use_num );

#ifdef _TWOP_STATS
      /**********************************************************
       * 
       * STATISTICAL ANALYSIS
       * 
       **********************************************************/
      for ( int ireim = 0; ireim < 1; ireim++ ) {

        if ( num_conf < 6 ) {
          fprintf ( stderr, "[xg_exdefl_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
          EXIT(1);
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it1 = 0; it1 < T_global; it1++ ) {
          for ( int it2 = 0; it2 < T_global; it2++ ) {
            int const idt = ( it2 - it1 + T_global ) % T_global;

            data[iconf][idt] += twop_orbit[iconf][inev][it1][2*it2+ireim];
          }}
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] /= (double)T_global;
          }
        }

        char obs_name[200];
        sprintf( obs_name, "%s.%s", obs_name_prefix, reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        if ( write_data == 1 ) {
          sprintf ( filename, "%s.corr", obs_name_prefix );
          FILE * ofs = fopen ( filename, "w" );
 
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            fprintf( ofs, "#  %c %6d\n", conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( ofs, "%25.16e%25.16e\n", twop_orbit[iconf][2*it], twop_orbit[iconf][2*it+1] );
            }
          }
          fclose ( ofs );
        }  /* end of if write data */

        /**********************************************************
         * acosh ratio for m_eff
         **********************************************************/
        int const Thp1 = T_global / 2 + 1;
        for ( int itau = 1; itau < Thp1/2; itau++ ) {
          int narg = 3;
          int arg_first[3] = { 0, 2 * itau, itau };
          int arg_stride[3] = {1,1,1};
          int nT = Thp1 - 2 * itau;

          sprintf ( obs_name, "%s.acosh_ratio.tau%d.%s", obs_name_prefix, itau, reim_str[ireim] );

          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
        }  /* end of loop on tau */

        fini_2level_dtable ( &data );
      }  /* end of loop on reim */

    }  /* end of loop on evecs use accum steps */

#endif  /* of ifdef _TWOP_STATS */

#ifdef _LOOP_ANALYSIS
    /**********************************************************
     *
     * LOOP ANALYSIS AND 3PT
     *
     **********************************************************/

    /**********************************************************
     * loop on stout smearing levels
     **********************************************************/
  for ( unsigned int istout = 0; istout < stout_level_num; istout++ ) {

    /**********************************************************
     * loop field
     **********************************************************/
    double *** loop = NULL;
  
    loop = init_3level_dtable ( num_conf, T_global, 2 );
    if ( loop == NULL ) {
      fprintf ( stdout, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(25);
    }
  
    /**********************************************************
     *
     * read loop data
     *
     **********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    
      /***********************************************************
       * open AFF reader
       ***********************************************************/
      struct AffNode_s *affn = NULL, *affdir = NULL;
  
      sprintf ( filename, "%s/stream_%c/%d/cpff.xg.%d.aff", filename_prefix3, conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], conf_src_list[iconf][0][1] );
  
      fprintf(stdout, "# [xg_exdefl_analyse] reading data from file %s\n", filename);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
  
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }
  
  
      sprintf( key, "/StoutN%d/StoutRho%6.4f", stout_level_iter[istout], stout_level_rho[istout] );
          
      if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_exdefl_analyse] key = %s\n", key );
  
      affdir = aff_reader_chpath (affr, affn, key );
      if( affdir == NULL ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
        EXIT(105);
      }
  
      uint32_t uitems = 2 * T_global;
      exitstatus = aff_node_get_double ( affr, affdir, loop[iconf][0], uitems );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[xg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(105);
      }
  
      aff_reader_close ( affr );

    }  /* end of loop on configs */
  
    char smearing_tag[50];
    sprintf ( smearing_tag, "stout_%d_%6.4f", stout_level_iter[istout], stout_level_rho[istout] );

    /**********************************************************
     * analyse plaquettes
     **********************************************************/
    {

      double ** data = init_2level_dtable ( num_conf, 4 );
      if ( data == NULL ) {
        fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(25);
      }

#pragma omp parallel for
      for ( int i = 0; i< num_conf; i++ ) {
        data[i][0] = 0.;
        data[i][1] = 0.;
        data[i][2] = 0.;
        data[i][3] = 0.;
        for( int it = 0; it < T_global; it++ ) {
          data[i][0] += loop[i][it][0];
          data[i][1] += loop[i][it][1];
          data[i][2] += loop[i][it][0] + loop[i][it][1];
          data[i][3] += loop[i][it][0] - loop[i][it][1];
        }
        data[i][0] /= (18. * VOLUME);
        data[i][1] /= (18. * VOLUME);
        data[i][2] /= (18. * VOLUME);
        data[i][3] /= (18. * VOLUME);
      }

      char obs_name[100];
      sprintf ( obs_name, "plaquette.%s" , smearing_tag );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, 4, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      fini_2level_dtable ( &data );

    }  /* end of block */

    /**********************************************************
     *
     * build trace-subtracted tensor
     *
     **********************************************************/
    double ** loop_sub = init_2level_dtable ( num_conf, T_global );
    if ( loop_sub == NULL ) {
      fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(25);
    }
  
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int it = 0; it < T_global; it++ ) {
        loop_sub[iconf][it] = loop[iconf][it][0] - loop[iconf][it][1];
      }
    }  /* end of loop on configs */
  
    fini_3level_dtable ( &loop );
  
    /**********************************************************
     * tag to characterize the loops w.r.t. low-mode and
     * stochastic part
     **********************************************************/
    /**********************************************************
     * write loop_sub to separate ascii file
     **********************************************************/
  
    if ( write_data ) {
      sprintf ( filename, "loop_sub.%s.corr", smearing_tag );
  
      FILE * loop_sub_fs = fopen( filename, "w" );
      if ( loop_sub_fs == NULL ) {
        fprintf ( stderr, "[xg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      } 
  
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        fprintf ( loop_sub_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
        for ( int it = 0; it < T_global; it++ ) {
          fprintf ( loop_sub_fs , "%25.16e\n", loop_sub[iconf][it] );
        }
      }
      fclose ( loop_sub_fs );
    }  /* end of if write data */
  
    /**********************************************************
     *
     * STATISTICAL ANALYSIS OF LOOP VEC
     *
     **********************************************************/
  
    if ( num_conf < 6 ) {
      fprintf ( stderr, "[xg_exdefl_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }
  
    char obs_name[100];
    sprintf ( obs_name, "loop_sub.%s" , smearing_tag );
  
    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( loop_sub[0], num_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
  
#ifdef _RAT_METHOD
    /**********************************************************
     *
     * STATISTICAL ANALYSIS for products and ratios
     *
     * fixed source - sink separation
     *
     **********************************************************/
  
    /**********************************************************
     * loop on number of evecs
     **********************************************************/
    for ( int inev = 0; inev  < evecs_use_nstep ; inev++ ) {

      int const evecs_use_num = evecs_use_min + inev * evecs_use_step;

      /**********************************************************
       * loop on source - sink time separations
       **********************************************************/
      for ( int idt = 0; idt <= T_global/2; idt++ )
      {
    
        double *** threep_44 = init_3level_dtable ( num_conf, T_global, T_global ) ;
        if ( threep_44 == NULL ) {
          fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
    
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int tsrc = 0; tsrc < T_global; tsrc++ ) {
            
            int const tsnk1 = ( tsrc + idt + T_global ) % T_global;
            int const tsnk2 = ( tsrc - idt + T_global ) % T_global;
    
            /**********************************************************
             * !!! LOOK OUT:
             *       This includes the momentum orbit average !!!
             **********************************************************/
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    
              const double a[2] = { twop[imom][iconf][inev][tsrc][2*(tsnk1)  ], twop[imom][iconf][inev][tsrc][2*(tsnk1)+1] };
              const double b[2] = { twop[imom][iconf][inev][tsrc][2*(tsnk2)  ], twop[imom][iconf][inev][tsrc][2*(tsnk2)+1] };

              /**********************************************************
               * loop relative insertion times; relative to source time
               **********************************************************/
              for ( int it = 0; it < T_global; it++ ) {
    
                /**********************************************************
                 * absolute insertion times
                 **********************************************************/
                int const tins1 = (  tsrc + it + T_global ) % T_global;
                int const tins2 = (  tsrc - it + T_global ) % T_global;
    
                if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_average] insertion times tsrc %3d    dt %3d    tc %3d    tins %3d %3d\n", tsrc, idt, it, tins1, tins2 );
    
                /**********************************************************
                 * O44, real parts only
                 **********************************************************/

                threep_44[iconf][tsrc][it] += ( a[0] * loop_sub[iconf][tins1] + b[0] * loop_sub[iconf][tins2] ) * 0.5;
    
              }  /* end of loop on it */
    
            }  /* end of loop on imom */
    
            /**********************************************************
             * normalize
             **********************************************************/
            /* O44 simple orbit average */
            double const norm44 = 1. / g_sink_momentum_number;
            for ( int it = 0; it < T_global; it++ ) {
              threep_44[iconf][tsrc][it] *= norm44;
            }
          }  /* end of loop on isrc */
        }  /* end of loop on iconf */
    
        char obsname_prefix[400];
        sprintf ( obsname_prefix, "threep.gf_%s.gi_%s.g4_D4.%s.dt%d.px%d_py%d_pz%d.nev%d",
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            smearing_tag, idt,
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], evecs_use_num );

        /**********************************************************
         * write 3pt function to ascii file, per source
         **********************************************************/
        if ( write_data == 1) {
          /**********************************************************
           * write 44 3pt
           **********************************************************/
          sprintf ( filename, "%s.%s.corr", obsname_prefix, reim_str[0] );
          FILE *ofs = fopen( filename , "w" );
    
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < T_global; isrc++ ) {
            fprintf ( ofs, "# %c %6d %3d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], isrc );
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( ofs, "%25.16e\n", threep_44[iconf][isrc][it] );
            }
          }}
          fclose ( ofs );
        }  /* end of if write_data */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for threep
         *
         * with fixed source - sink separation
         *
         **********************************************************/
    
        for ( int ireim = 0; ireim < 1; ireim++ ) {

          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * threep_44
           **********************************************************/
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < T_global; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it];
              }
              data[iconf][it] = dtmp / (double)T_global;
            }
          }
    
          char obs_name[100];
          sprintf ( obs_name, "%s.src-avg.%s", obsname_prefix, reim_str[ireim] );
    
          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          if ( write_data == 1 ) {
            sprintf ( filename, "%s.corr", obs_name );
            FILE *ofs = fopen( filename , "w" );

            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              fprintf ( ofs, "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
              for ( int it = 0; it < T_global; it++ ) {
                fprintf ( ofs, "%25.16e\n", data[iconf][it] );
              }
            }
            fclose ( ofs );
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for ratio 
         *   with source - sink fixed
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          /* UWerr parameters */
          int narg          = 2;
          int arg_first[2]  = { 0, T_global };
          int arg_stride[2] = { 1,  0 };
          char obs_name[100];
    
          double ** data = init_2level_dtable ( num_conf, T_global + 1 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * O44, source average
           * TWOP, orbit-averaged, source average
           **********************************************************/
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < T_global; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it];
              }
              data[iconf][it] = dtmp / (double)T_global;
            }
          
            for ( int isrc = 0; isrc < T_global; isrc++ ) {

              int const tsink1 = ( isrc + idt + T_global ) % T_global;
              int const tsink2 = ( isrc - idt + T_global ) % T_global;

              data[iconf][T_global] += ( twop_orbit[iconf][inev][isrc][2*tsink1] + twop_orbit[iconf][inev][isrc][2*tsink2] );
            }
            data[iconf][T_global] /= (double)T_global;
          }

          sprintf ( obs_name, "ratio.orbit.src-avg.gf_%s.gi_%s.g4_D4.%s.dt%d.PX%d_PY%d_PZ%d.nev%d.%s",
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            smearing_tag, idt,
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], evecs_use_num, reim_str[ireim] );
    
          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global + 1, T_global, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for ratio
         * with per-time-slice subtraction
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {

          double ** data = init_2level_dtable ( num_conf, ( 2 * T_global + 1 ) * T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_exdefl_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              for ( int isrc = 0; isrc < T_global; isrc++ ) {
                data[iconf][it*T_global + isrc] = threep_44[iconf][isrc][it];
              }
            }
            for ( int it = 0; it < T_global; it++ ) {
              for ( int isrc = 0; isrc < T_global; isrc++ ) {
                int const tins = ( isrc + it + T_global ) % T_global;
                data[iconf][ T_global * T_global + it * T_global ] =  loop_sub[iconf][tins];
              }
            }
            for ( int isrc = 0; isrc < T_global; isrc++ ) {
              int const tsink = ( isrc + idt + T_global ) % T_global;
              data[iconf][ 2 * T_global * T_global + isrc] = twop_orbit[iconf][inev][isrc][2*tsink];
            }
          }

          /* UWerr parameters */
          int narg          = ( 2 * T_global + 1 ) * T_global;
          int * arg_first = init_1level_itable ( narg );
          int arg_first[2]  = { 0, T_global };

          int * arg_stride = init_1level_itable ( narg );
          for ( int i = 0; i < T_global; i++ ) arg_stride[i] = 1;

          char obs_name[100];


          fini_2level_dtable ( &data );
          fini_1level_dtable ( &arg_first );
          fini_1level_dtable ( &arg_stride );
        }

        fini_3level_dtable ( &threep_44 );
    
      }  /* end of loop on dt */

    }  /* end of loop on evecs use */

#endif  /* end of ifdef _RAT_METHOD */
  
    /**********************************************************/
    /**********************************************************/
  
    fini_2level_dtable ( &loop_sub );


  }  /* end of loop on smearing levels */
#endif  /* end of ifdef _LOOP_ANALYSIS */

    fini_5level_dtable ( &twop );
    fini_4level_dtable ( &twop_orbit );

  }}  /* end of loop on gi, gf */

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [xg_exdefl_analyse] %s# [xg_exdefl_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [xg_exdefl_analyse] %s# [xg_exdefl_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
