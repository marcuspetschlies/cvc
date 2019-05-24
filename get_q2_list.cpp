/****************************************************
 * get_q2_list
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"

#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

typedef struct {
  char name[400];
  double beta;
  double a_fm;
  double ainv_gev;
  unsigned int L;
  double ampi;
  double dampi;
} ensemble_data_struct;

/***************************************************************************
 *
 ***************************************************************************/
int init_ensemble_data ( ensemble_data_struct * p , char * name ) {

  sprintf ( p->name, "NA" );
  p->beta = 0.;
  p->a_fm = 0.;
  p->ainv_gev = 0.;
  p->L = 0;
  p->ampi = 0.;
  p->dampi = 0.;
 
  if ( strcmp ( name , "cA211.53.24") == 0 ) {
    p->beta = 1.726; p->a_fm = 0.098; p->L = 24; p->ampi = 0.1667; p->dampi = 0.0006;
  } else if ( strcmp ( name, "cA211.40.24" ) == 0 ) {
    p->beta = 1.726; p->a_fm = 0.098; p->L = 24; p->ampi = 0.1452; p->dampi = 0.0006;
  } else if ( strcmp ( name, "cA211.30.32" ) == 0 ) {
    p->beta = 1.726; p->a_fm = 0.098; p->L = 24; p->ampi = 0.1256; p->dampi = 0.0003;
  } else if ( strcmp ( name, "cA211.12.48" ) == 0 ) {
    p->beta = 1.726; p->a_fm = 0.098; p->L = 24; p->ampi = 0.0794; p->dampi = 0.0002;
  } else if ( strcmp ( name, "cB211.25.48" ) == 0 ) {
    p->beta = 1.778; p->a_fm = 0.083; p->L = 48; p->ampi = 0.1042; p->dampi = 0.0002;
  } else if ( strcmp ( name, "cB211.072.64" ) == 0 ) {
    p->beta = 1.778; p->a_fm = 0.083; p->L = 64; p->ampi = 0.0566; p->dampi = 0.0002;
  } else if ( strcmp ( name, "cC211.060.80" ) == 0 ) {
    p->beta = 1.836; p->a_fm = 0.071; p->L = 80; p->ampi = 0.0474; p->dampi = 0.0002;
  } else if ( strcmp ( name, "test" ) == 0 ) {
    p->beta = 1.; p->a_fm = 0.1; p->L = 4; p->ampi = 0.1; p->dampi = 0.01;
  } else {
    fprintf ( stderr, "[init_ensemble_data] Error, unknown ensemble name\n" );
    return ( 1 );
  }

  strcpy ( p->name , name );
  p->ainv_gev = 1. / p->a_fm * 0.1973269631;

  return ( 0 );
}  /* end of init_ensemble_data */

/***************************************************************************/
/***************************************************************************/

double qsqr_onshell ( double const m1, double const  m2, double const p1[3], double const p2[3] )  {

  double const E1 = sqrt ( m1 * m1 +  p1[0] * p1[0]  + p1[1] * p1[1]  + p1[2] * p1[2] );
  double const E2 = sqrt ( m2 * m2 +  p2[0] * p2[0]  + p2[1] * p2[1]  + p2[2] * p2[2] );

  return( m1 * m1 + m2 * m2 - 2. * ( E1 * E2 - ( p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2] ) ) );

}  /* end of qsqr_onshell */

/***************************************************************************/
/***************************************************************************/

int main(int argc, char **argv) {

  double const _Q2_EPS = 1.e-14;

  const char outfile_prefix[] = "get_q2_list";

  int c;
  int exitstatus;
  int io_proc = 2;
  // double ratime, retime;
  int pn_cut[3] = {-1, -1, -1};

  double q2_cut = 0.;

  char ensemble_name[100];

  while ((c = getopt(argc, argv, "h?N:C:q:")) != -1) {
    switch (c) {
    case 'N':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [get_q2_list] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'C':
      sscanf ( optarg, "%d,%d,%d", pn_cut, pn_cut+1, pn_cut+2 );
      fprintf ( stdout, "# [get_q2_list] pn_cut set to %3d %3d %3d\n", pn_cut[0], pn_cut[1], pn_cut[2] );
      break;
    case 'q':
      q2_cut = atof ( optarg );
      fprintf ( stdout, "# [get_q2_list] q2_cut set to %e\n", q2_cut );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);
  fprintf(stdout, "# [get_q2_list] %s# [get_q2_list] start of run\n", ctime(&g_the_time));
  fprintf(stderr, "# [get_q2_list] %s# [get_q2_list] start of run\n", ctime(&g_the_time));

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [get_q2_list] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * set ensemble data
   ***************************************************************************/
  ensemble_data_struct ens;
  init_ensemble_data ( &ens, ensemble_name );

  double const PUNIT = 2. * M_PI / (double)ens.L;
  int const Lh = ens.L / 2;

  int ** p_list = init_2level_itable ( ens.L * ens.L * ens.L, 7 );

  int count = 0;
  for ( int i1 = 0; i1 < ens.L; i1++ ) {
    int const k1 = i1 > Lh ? i1 - ens.L : i1;

  for ( int i2 = 0; i2 < ens.L; i2++ ) {
    int const k2 = i2 > Lh ? i2 - ens.L : i2;

  for ( int i3 = 0; i3 < ens.L; i3++ ) {
    int const k3 = i3 > Lh ? i3 - ens.L : i3;
  
    p_list[count][0] = k1;
    p_list[count][1] = k2;
    p_list[count][2] = k3;

    p_list[count][3] = k1 * k1 + k2 * k2 + k3 * k3;
    p_list[count][4] = k1 * k1 * k1 * k1 + k2 * k2 * k2 * k2 + k3 * k3 * k3 * k3;
    p_list[count][5] = k1 * k1 * k1 * k1 * k1 * k1 + + k2 * k2 * k2 * k2 * k2 * k2 + k3 * k3 * k3 * k3 * k3 * k3;

    p_list[count][6] = 
      ( pn_cut[0] < 0 || p_list[count][3] <= pn_cut[0] ) && 
      ( pn_cut[1] < 0 || p_list[count][4] <= pn_cut[1] ) && 
      ( pn_cut[2] < 0 || p_list[count][5] <= pn_cut[2] ) ;

    count++;
  }}}

  FILE * ofs = NULL;
  char filename[400];
  sprintf ( filename, "%s.%s", outfile_prefix, ens.name );
  ofs = fopen ( filename, "w" );

  fprintf ( ofs, "# name     = %s\n", ens.name );
  fprintf ( ofs, "# L        = %u\n", ens.L );
  fprintf ( ofs, "# a_fm     = %e\n", ens.a_fm );
  fprintf ( ofs, "# ainv_gev = %e\n", ens.ainv_gev );
  fprintf ( ofs, "# ampi     = %e\n", ens.ampi );
  fprintf ( ofs, "# dampi    = %e\n", ens.dampi );
  fprintf ( ofs, "# pn_cut   = %3d %3d %3d\n", pn_cut[0], pn_cut[1], pn_cut[2] );
  fprintf ( ofs, "# q2_cut   = %e\n#\n", q2_cut );

  for ( int i = 0; i < ens.L * ens.L * ens.L; i++ ) {
    if ( p_list[i][6] == 0 ) continue;

    double const p1[3] = {
      p_list[i][0] * PUNIT,
      p_list[i][1] * PUNIT,
      p_list[i][2] * PUNIT };

    for ( int k = 0; k < ens.L * ens.L * ens.L; k++ ) {
      if ( p_list[k][6] == 0 ) continue;

      double const p2[3] = {
        p_list[k][0] * PUNIT,
        p_list[k][1] * PUNIT,
        p_list[k][2] * PUNIT };


      double q2_gev2 = qsqr_onshell ( ens.ampi, ens.ampi, p1, p2 ) * ens.ainv_gev * ens.ainv_gev;

      if ( q2_cut > 0. && fabs( q2_gev2 ) > q2_cut ) continue;

      int const pp = p_list[i][0] * p_list[k][0] + p_list[i][1] * p_list[k][1] + p_list[i][2] * p_list[k][2]; 

      if ( fabs( q2_gev2 ) < _Q2_EPS ) q2_gev2 = 0.;

      fprintf ( ofs, "%3d %3d %3d   %10d %10d %10d     %3d %3d %3d   %10d %10d %10d   %10d   %16.7e\n", 
          p_list[i][0], p_list[i][1], p_list[i][2], p_list[i][3], p_list[i][4], p_list[i][5],
          p_list[k][0], p_list[k][1], p_list[k][2], p_list[k][3], p_list[k][4], p_list[k][5],
          pp, q2_gev2 );
    }
  }


  fclose ( ofs );



  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  fini_2level_itable ( &p_list );

  g_the_time = time(NULL);
  fprintf(stdout, "# [get_q2_list] %s# [get_q2_list] end of run\n", ctime(&g_the_time));
  fprintf(stderr, "# [get_q2_list] %s# [get_q2_list] end of run\n", ctime(&g_the_time));

  return(0);

}
