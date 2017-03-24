/****************************************************
 * test_sliced.cpp 
 *
 * Do 30. Jun 08:19:01 CEST 2016
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
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
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

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
#include "scalar_products.h"


using namespace cvc;

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int i, j;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int ix_even, ix_odd, y0, y1, y2, y3;
  /* int start_valuet=0, start_valuex=0, start_valuey=0; */
  /* int threadid, nthreads; */
  /* double diff1, diff2; */
  double plaq=0;
  double spinor1[24], spinor2[24];
  complex w, w2;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double norm, norm2;
  unsigned int Vhalf, VOL3, VOL3half;
  double **eo_spinor_field = NULL;
  int no_eo_fields;


#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "apply_Dtm.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);


  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "Error from init_geometry\n");
    exit(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  VOL3 = LX * LY * LZ;
  VOL3half = VOL3 / 2;

#if 0
  for(ix=0; ix<Vhalf; ix++) {
    x0 = g_eosub2t[0][ix];
    fprintf(stdout, "proc%.4d e2sliced\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_eosub2sliced3d[0][ix]);
  }

  for(ix=0; ix<Vhalf; ix++) {
    x0 = g_eosub2t[1][ix];
    fprintf(stdout, "proc%.4d o2sliced\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_eosub2sliced3d[1][ix]);
  }

  for(x0=0; x0<T; x0++) {
    for(ix=0; ix<VOL3/2; ix++) {
      fprintf(stdout, "proc%.4d sliced2e\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_sliced3d2eosub[0][x0][ix]);
    }
  }

  for(x0=0; x0<T; x0++) {
    for(ix=0; ix<VOL3/2; ix++) {
      fprintf(stdout, "proc%.4d sliced2o\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_sliced3d2eosub[1][x0][ix]);
    }
  }
#endif  /* of if 0 */

  size_t sizeof_eo_spinor_field = 12 * VOLUME * sizeof(double);
  size_t sizeof_eo_spinor_field_timeslice = 12 * VOL3 * sizeof(double);

  int tseq = 3;

  int source_proc_id = tseq / T == g_proc_coords[0] ? g_cart_id : -1;
  int tloc = tseq % T;

  double *s_even = (double*)malloc(sizeof_eo_spinor_field);
  double *s_odd  = (double*)malloc(sizeof_eo_spinor_field);
  double *p_even = (double*)malloc(sizeof_eo_spinor_field);
  double *p_odd  = (double*)malloc(sizeof_eo_spinor_field);
  double *t_even = (double*)malloc(sizeof_eo_spinor_field);
  double *t_odd  = (double*)malloc(sizeof_eo_spinor_field);

  random_spinor_field (p_even, Vhalf);
  random_spinor_field (p_odd, Vhalf);


  const double q[3] = {
    2.*M_PI * 0 / (LX_global),
    2.*M_PI * 0 / (LY_global),
    2.*M_PI * 0 / (LZ_global) };

  const double  q_offset = q[0] * g_proc_coords[1] * LX + q[1] * g_proc_coords[2] * LY + q[2] * g_proc_coords[3] * LZ;
  const int gseq = 14;
  const size_t offset = _GSI(VOL3half);

  if( source_proc_id == g_cart_id ) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[tloc][x1][x2][x3];
      double q_phase = q[0] * x1 + q[1] * x2 + q[2] * x3 + q_offset;
      w.re = cos(q_phase);
      w.im = sin(q_phase);
      double *s_, *p_;
      if(g_iseven[ix]) {
        s_ = s_even + _GSI(g_lexic2eosub[ix]);
        p_ = p_even + _GSI(g_lexic2eosub[ix]);
      } else {
        s_ = s_odd + _GSI(g_lexic2eosub[ix]);
        p_ = p_odd + _GSI(g_lexic2eosub[ix]);
      }
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(spinor2, gseq, spinor1);
      _fv_eq_gamma_ti_fv(s_, 5, spinor2);
    }}}
    for(i=1; i<T; i++) {
      x0 = (tloc + i) % T;
      memset(s_even+x0*offset, 0, sizeof_eo_spinor_field_timeslice);
      memset(s_odd +x0*offset, 0, sizeof_eo_spinor_field_timeslice);
    }
  } else {
    // printf("# [] process %d setting source to zero\n", g_cart_id);
    memset(s_even, 0, sizeof_eo_spinor_field);
    memset(s_odd,  0, sizeof_eo_spinor_field);
  }
                                                                      
  if( source_proc_id == g_cart_id ) {
    /* even part */
    for(ix = 0; ix<VOL3half; ix++ ) {
      unsigned ixeosub = tloc*VOL3half+ix;
      double q_phase = q[0] * g_eosubt2coords[0][tloc][ix][0] + q[1] * g_eosubt2coords[0][tloc][ix][1] + q[2] * g_eosubt2coords[0][tloc][ix][2] + q_offset;
      complex w = {cos(q_phase), sin(q_phase) };
      double *s_ = t_even + _GSI(ixeosub);
      double *p_ = p_even + _GSI(ixeosub);
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(spinor2, gseq, spinor1);
      _fv_eq_gamma_ti_fv(s_, 5, spinor2);
    }
    /* odd part */
    for(ix = 0; ix<VOL3half; ix++ ) {
      unsigned ixeosub = tloc*VOL3half+ix;
      double q_phase = q[0] * g_eosubt2coords[1][tloc][ix][0] + q[1] * g_eosubt2coords[1][tloc][ix][1] + q[2] * g_eosubt2coords[1][tloc][ix][2] + q_offset;
      complex w = {cos(q_phase), sin(q_phase) };
      double *s_ = t_odd + _GSI(ixeosub);
      double *p_ = p_odd + _GSI(ixeosub);
      _fv_eq_fv_ti_co(spinor1, p_, &w);
      _fv_eq_gamma_ti_fv(spinor2, gseq, spinor1);
      _fv_eq_gamma_ti_fv(s_, 5, spinor2);
    }
    for(i=1; i<T; i++) {
      x0 = (tloc + i) % T;
      memset(t_even+x0*offset, 0, sizeof_eo_spinor_field_timeslice);
      memset(t_odd +x0*offset, 0, sizeof_eo_spinor_field_timeslice);
    }
  } else {
    memset(t_even, 0, sizeof_eo_spinor_field);
    memset(t_odd,  0, sizeof_eo_spinor_field);
  }

  spinor_field_norm_diff (&norm, s_even, t_even, Vhalf );
  spinor_field_norm_diff (&norm2, s_odd, t_odd, Vhalf );

  if( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_sliced] even part %e\n", norm);
    fprintf(stdout, "# [test_sliced] odd  part %e\n", norm2);
  }



  free(s_even);
  free(s_odd);
  free(p_even);
  free(p_odd);
  free(t_even);
  free(t_odd);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free_geometry();

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}
