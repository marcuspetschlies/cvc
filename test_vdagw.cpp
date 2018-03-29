/****************************************************
 * test_vdagw.cpp
 *
 * Do 22. Mär 12:56:24 CET 2018
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "set_default.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "Q_phi.h"
#include "scalar_products.h"
#include "ranlxd.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "vdag_w_utils.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  char filename[200];


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
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
  if(g_cart_id==0) fprintf(stdout, "# [test_vdagw] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if( exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if( exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if( exitstatus != 0) {
    EXIT(16);
  }
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_vdagw] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();


  int const io_proc = get_io_proc ();

  int exitstatus;
  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3half = LX*LY*LZ / 2;

  unsigned int const nv = 11;
  unsigned int const nw =  7;

  double *** contr  = init_3level_dtable ( nv, nw, 2*VOL3half );
  double *** contr2 = init_3level_dtable ( nv, nw, 2*VOL3half );

  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

#if 0

  double ** v = init_2level_dtable ( nv, _GSI( (VOLUME+RAND)/2 ) );
  double ** w = init_2level_dtable ( nw, _GSI( (VOLUME+RAND)/2 ) );

  /* set the spinor field */
  for ( unsigned int i = 0; i < nv; i++ ) {
    rangauss ( v[i], _GSI(Vhalf) );
#ifdef HAVE_MPI
    xchange_eo_field( v[i], 0 );
#endif
  }

  for ( unsigned int i = 0; i < nw; i++ ) {
    rangauss ( w[i], _GSI(Vhalf) );
#ifdef HAVE_MPI
    xchange_eo_field( w[i], 0 );
#endif
  }


  for ( int t = 0; t <= T; t++ ) {
    exitstatus = vdag_w_spin_color_reduction ( contr, v, w, nv, nw, t );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_vdagw] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(102);
    }

    for ( unsigned int i = 0; i < nv; i++ ) {

    for ( unsigned int k = 0; k < nw; k++ ) {

      double norm = 0.;
      for ( int x1 = 0; x1 < LX; x1++ ) {
      for ( int x2 = 0; x2 < LY; x2++ ) {
      for ( int x3 = 0; x3 < LZ; x3++ ) {

        unsigned int const ix = t==T ? g_iup[g_ipt[t-1][x1][x2][x3]][0] : g_ipt[t][x1][x2][x3];
        unsigned int const ixeosub = g_lexic2eosub[ix];
        if ( ! g_iseven[ix] ) continue;
        unsigned int const iix = _GSI( ixeosub );
        complex z;
        _co_eq_fv_dag_ti_fv ( &z, v[i]+ iix, w[k] + iix );
        unsigned int const iz = g_lexic2eosub[ g_ipt[0][x1][x2][x3] ];
        double const dtmp[2] = {
          z.re - contr[i][k][2*iz ],
          z.im - contr[i][k][2*iz+1]
        };
        norm += dtmp[0] * dtmp[0] + dtmp[1] * dtmp[1];

        // if ( t == T ) 
        //  fprintf ( stdout, "# [test_vdagw] proc%.4d x %3d %3d %3d %3d i %2d k %2d z %25.16e %25.16e c %25.16e %25.16e\n", g_cart_id, 
        //    t + g_proc_coords[0]*T, x1 + g_proc_coords[1]*LX, x2 + g_proc_coords[2]*LY, x3 + g_proc_coords[3]*LZ,
        //    i, k, z.re, z.im, contr[i][k][2*iz ], contr[i][k][2*iz+1] );
      }}}

      norm = sqrt ( norm );

      fprintf ( stdout, "# [test_vdagw] proc%.4d t %3d i %2d k %2d diff %25.16e\n", g_cart_id, t + g_proc_coords[0]*T, i, k, norm );

    }}
  }
  fini_2level_dtable ( &v );
  fini_2level_dtable ( &w );

#endif  // of if 0

  rangauss ( contr[0][0], nv*nw* 2 * VOL3half );
  rangauss ( contr2[0][0], nv*nw* 2 * VOL3half );

  for ( int mu = 0; mu < 4; mu++ ) {
    int s[3] = {0,0,0};
    if ( mu > 0 ) s[mu-1] = -1;

    for ( int ieo = 0; ieo < 2; ieo++ ) {
      if ( g_cart_id == 0 ) fprintf ( stdout, "# [test_vdagw] mu %d ieo %2d\n", mu, ieo );
#ifdef HAVE_MPI
      MPI_Barrier ( g_cart_grid );
#endif
      
      for ( int t = 0; t < T; t++ ) {
  
        double _Complex *** contr_p = init_3level_ztable ( g_sink_momentum_number, nv, nw );
        double _Complex *** contr_p2 = init_3level_ztable ( g_sink_momentum_number, nv, nw );

        exitstatus = vdag_w_momentum_projection ( contr_p, contr, nv, nw, g_sink_momentum_list, g_sink_momentum_number, t, ieo, mu, 0);
        exitstatus = vdag_w_momentum_projection ( contr_p, contr2, nv, nw, g_sink_momentum_list, g_sink_momentum_number, t, ieo, mu, 1 );

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

          double const p[3] = {
            2.*M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
            2.*M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
            2.*M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
          };
          double const phase_offset = g_proc_coords[1]*LX * p[0] + g_proc_coords[2]*LY * p[1] + g_proc_coords[3]*LZ * p[2];

          unsigned int ixeo = 0;
          for ( int x1 = 0; x1 < LX; x1++ ) {
          for ( int x2 = 0; x2 < LY; x2++ ) {
          for ( int x3 = 0; x3 < LZ; x3++ ) {

            int const r[3] =  { x1, x2, x3 };

            unsigned int const ix = g_ipt[t][x1][x2][x3];
            if ( !g_iseven[ix] != ieo ) continue;
           
            double const phase = phase_offset + ( r[0] + s[0] ) * p[0] + ( r[1] + s[1] ) * p[1] + ( r[2] + s[2] ) * p[2];
            double _Complex const ephase = cexp ( I*phase );

            for ( unsigned int iv = 0; iv < nv; iv++ ) {
            for ( unsigned int iw = 0; iw < nw; iw++ ) {
              contr_p2[imom][iv][iw] += ( contr[iv][iw][2*ixeo] + I * contr[iv][iw][2*ixeo+1] ) * ephase;
              contr_p2[imom][iv][iw] += ( contr2[iv][iw][2*ixeo] + I * contr2[iv][iw][2*ixeo+1] ) * ephase;
            }}

            ixeo++;
          }}}
        }  // end of loop on momenta


#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
        void *buffer = malloc ( nv*nw*g_sink_momentum_number * 2 *sizeof (double) );
        memcpy ( buffer,  contr_p2[0][0], nv*nw*g_sink_momentum_number * 2 * sizeof(double) );
        MPI_Allreduce( buffer, contr_p2[0][0], 2*nv*nw*g_sink_momentum_number, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        free ( buffer );
#  endif
#endif
#if 0
#endif  // of if 0

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          double const eps = 1.e-14;
          for ( unsigned int iv = 0; iv < nv; iv++ ) {
            for ( unsigned int iw = 0; iw < nw; iw++ ) {
              double norm = 
                cabs ( contr_p[imom][iv][iw] - contr_p2[imom][iv][iw] ) / ( 0.5 * cabs ( contr_p[imom][iv][iw] + contr_p2[imom][iv][iw] ) );
              if ( io_proc >= 1 ) { 
                fprintf ( stdout, " t %3d p %3d %3d %3d v %2d w %2d  c %25.16e %25.16e c2 %25.16e %25.16e  diff %25.16e\n", t + g_proc_coords[0]*T, 
                    g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                    iv, iw, 
                    creal ( contr_p[imom][iv][iw]  ), cimag ( contr_p[imom][iv][iw]  ), creal ( contr_p2[imom][iv][iw]  ), cimag ( contr_p2[imom][iv][iw]  ),
                    norm );
                if ( norm > eps ) 
                  fprintf ( stderr, "[test_vdagw] Error for t %3d p %3d %3d %3d v %2d w %2d diff %25.16e\n", t + g_proc_coords[0]*T, 
                      g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                      iv, iw, norm );
                  
              }
            }
          }
        }

        fini_3level_ztable ( &contr_p );
        fini_3level_ztable ( &contr_p2 );
      }  // end of loop on ieo

    }  // end of loop on mu

  }  // end of loop on timeslices
    

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  fini_3level_dtable ( &contr );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_vdagw] %s# [test_vdagw] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_vdagw] %s# [test_vdagw] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

