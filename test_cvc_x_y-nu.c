/*******************************************************************
*test_cvc_x_y-nu.c
*August, 17th, 2011
*******************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#  include <omp.h>
#endif
#include "ifftw.h"
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

int main(int argc, char **argv) {
  
  int c, i, j, mu, nu, ir, is, ia, ib, imunu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int source_location[4], have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3, sx0m1, sx1m1, sx2m1, sx3m1;
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0, check_momentum_space_WI=0;
  int num_threads = 1, nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int mms = 0, mass_id = -1;
  int outfile_prefix_set = 0;
  int ud_one_file=0;
  double phase[4];
  char filename[100], contype[400], outfile_prefix[400];
  double ratime, retime;
  double Usourcebuff[72], *Usource[4];
  
    /* set the default values */

  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);
  
  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
  }
  
  T            = T_global;
  Tstart       = 0;
  
  fprintf(stdout, "T=%d, LX=%d, LY=%d, LZ=%d\n", T, LX, LY, LZ); 
    /* initialise geometry 
   * (all contained in cvc_geometry.c, 
   * init_geometry also initialises gamma matrices) */

   if(init_geometry() != 0) {
   fprintf(stderr, "ERROR from init_geometry\n");
   exit(1);
   }

   geometry();
   
   fprintf(stdout, "Initialised geometry.\n");
   
     /* determine source coordinates, find out, if source_location is in this process */
  /* The way the nu-dependence is implemented now only works if MPI is not used --> 
     If we decide to use this for production, carefully think about MPI version!!*/
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
  
  // take care of periodicity of the lattice
  if(sx0>0) 
    sx0m1=sx0-1;
  else sx0m1=T-1;
  
  if(sx1>0) 
    sx1m1=sx1-1;
  else sx1m1=LX-1;
  
  if(sx2>0) 
    sx2m1=sx2-1;
  else sx2m1=LY-1;
  
  if(sx3>0) 
    sx3m1=sx3-1;
  else sx3m1=LZ-1;
    
/*  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;*/
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    fprintf(stdout, "y-0, y-1, y-2, y-3: (%3d,%3d,%3d,%3d)\n", sx0m1, sx1m1, sx2m1, sx3m1);
/*    source_location[0] = g_ipt[sx0m1][sx1][sx2][sx3];
    source_location[1] = g_ipt[sx0][sx1m1][sx2][sx3];
    source_location[2] = g_ipt[sx0][sx1][sx2m1][sx3];
    source_location[3] = g_ipt[sx0][sx1][sx2][sx3m1];*/
    
/*    _cm_eq_cm_ti_co(Usource[0], &cvc_gauge_field[_GGI(source_location[0],0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &cvc_gauge_field[_GGI(source_location[1],1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &cvc_gauge_field[_GGI(source_location[2],2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &cvc_gauge_field[_GGI(source_location[3],3)], &co_phase_up[3]);*/
  }
  
}