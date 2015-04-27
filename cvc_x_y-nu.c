/*************************************************************************
*cvc_x_y-nu.c
*August, 17th, 2011
*calculates Pi_mu nu(q) from Pi_mu nu(x, y-nu)
*TO DO: adapt checking Ward identities
**************************************************************************/
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

void usage() {
  fprintf(stdout, "Code to perform contractions for conn. contrib. to vac. pol. with point sources at y and (y-a nu)\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -a write data also in ascii format and not only in binary lime format\n");
  fprintf(stdout, "         -w check Ward Identity in position space\n");
  fprintf(stdout, "         -W check Ward Identity in momentum space\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

void apply_gt_prop2(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location) {
#ifndef MPI
  int ix, ix1[5], k;
  double psi1[24], psi2[24], psi3[24], *work[3];
  complex co[3];
  char filename[200];

  /* allocate memory for work spinor fields */
  alloc_spinor_field(&work[0], VOLUME);
  alloc_spinor_field(&work[1], VOLUME);
  alloc_spinor_field(&work[2], VOLUME);

  if(format==0) {
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+0);
    read_lime_spinor(work[0], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+1);
    read_lime_spinor(work[1], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+2);
    read_lime_spinor(work[2], filename, 0);
  }
  else if(format==4) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+0);
    read_lime_spinor(work[0], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+1);
    read_lime_spinor(work[1], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+2);
    read_lime_spinor(work[2], filename, 0);
  }

  /* apply g to propagators from the left */
  for(k=0; k<3; k++) {
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_cm_ti_fv(psi1, &g[18*ix], &work[k][_GSI(ix)]);
      _fv_eq_fv(&work[k][_GSI(ix)], psi1);
    }
  }

  /* apply g to propagators from the right */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(psi1);
    _fv_eq_zero(psi2);
    _fv_eq_zero(psi3);

    if(mu==4) {
      co[0].re =  g[18*source_location + 6*ic+0];
      co[0].im = -g[18*source_location + 6*ic+1];
      co[1].re =  g[18*source_location + 6*ic+2];
      co[1].im = -g[18*source_location + 6*ic+3];
      co[2].re =  g[18*source_location + 6*ic+4];
      co[2].im = -g[18*source_location + 6*ic+5];
    }
    else {
      co[0].re =  g[18*g_idn[source_location][mu] + 6*ic+0];
      co[0].im = -g[18*g_idn[source_location][mu] + 6*ic+1];
      co[1].re =  g[18*g_idn[source_location][mu] + 6*ic+2];
      co[1].im = -g[18*g_idn[source_location][mu] + 6*ic+3];
      co[2].re =  g[18*g_idn[source_location][mu] + 6*ic+4];
      co[2].im = -g[18*g_idn[source_location][mu] + 6*ic+5];
    }

    _fv_eq_fv_ti_co(psi1, &work[0][_GSI(ix)], &co[0]);
    _fv_eq_fv_ti_co(psi2, &work[1][_GSI(ix)], &co[1]);
    _fv_eq_fv_ti_co(psi3, &work[2][_GSI(ix)], &co[2]);

    _fv_eq_fv_pl_fv(&phi[_GSI(ix)], psi1, psi2);
    _fv_pl_eq_fv(&phi[_GSI(ix)], psi3);
  }

  free(work[0]); free(work[1]); free(work[2]);
#endif
}


int main(int argc, char **argv) {
  
  int c, i, j, mu, nu, ir, is, ia, ib, imunu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int source_location[5], have_source_flag = 0;
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
  double gperm_sign[5][4], gperm2_sign[4][4];
  double *conn = (double*)NULL;
  double contact_term[8];
  double phase[4];
  double *work=NULL;
  int verbose = 0;
  int do_gt   = 0;
  char filename[100], contype[400], outfile_prefix[400];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  double *phi=NULL, *chi=NULL;
  complex w, w1;
  double Usourcebuff[72], *Usource[4]; 
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p;
  int *status;
#else
  fftwnd_plan plan_p;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif


  /* read the option arguments */
while ((c = getopt(argc, argv, "dwWah?vgf:t:m:o:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      fprintf(stdout, "\n# [] will check Ward identity in position space\n");
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf(stdout, "\n# [] will check Ward identity in momentum space\n");
      break;
    case 't':
      num_threads = atoi(optarg);
      fprintf(stdout, "\n# [] will use %d threads in spacetime loops\n", num_threads);
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [] will write data in ASCII format too\n");
      break;
    case 'm':
      mms = 1;
      mass_id = atoi(optarg);
      fprintf(stdout, "\n# [] will read propagators in MMS format with mass id %d\n", mass_id);
      break;
    case 'o':
      strcpy(outfile_prefix, optarg);
      fprintf(stdout, "\n# [] will use prefix %s for output filenames\n", outfile_prefix);
      outfile_prefix_set = 1;
      break;
    case 'd':
      ud_one_file = 1;
      fprintf(stdout, "\n# [] will take up- and down-propagator from same file\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* measure time */
  g_the_time = time(NULL);
  fprintf(stdout, "\n# cvc using global time stamp %s", ctime(&g_the_time));

  /* set number of openmp threads */
#ifdef OPENMP
  omp_set_num_threads(num_threads);
#endif

  /* set the default values */

  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);
#ifdef MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(7);
  }
#endif

  /* initialize fftw */
#ifdef OPENMP
  exitstatus = fftw_threads_init();
  if(exitstatus != 0) {
    fprintf(stderr, "\n[] Error from fftw_init_threads; status was %d\n", exitstatus);
    exit(120);
  }
#endif

  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
 
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif

  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

#ifdef MPI
  if(T==0) {
    fprintf(stderr, "[%2d] local T is zero; exit\n", g_cart_id);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(2);
  }
#endif

  /* initialise geometry 
   * (all contained in cvc_geometry.c, 
   * init_geometry also initialises gamma matrices) */

   if(init_geometry() != 0) {
   fprintf(stderr, "ERROR from init_geometry\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
   exit(1);
   }

   geometry();


  /* allocate memory for gauge field configuration
   * (contained in cvc_utils.c) */

  alloc_gauge_field(&cvc_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# cvc initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( cvc_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( cvc_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( cvc_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( cvc_gauge_field + _GGI(ix, 3) );
    }
  }

#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

   /* initialize gauge transformation */
   if(do_gt==1) {
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value after gauge trafo: %25.16e\n", plaq);
  }

  /* allocate memory for the spinor fields */
  no_fields = 24;
  if(mms) no_fields++;
  cvc_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&cvc_spinor_field[i], VOLUMEPLUSRAND);
  if(mms) {
    work = cvc_spinor_field[no_fields-1];
  }

  /* allocate memory for the contractions */
  conn = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( conn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(3);
  }
#ifdef OPENMP
#pragma omp parallel for
#endif
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] = 0.;

  /* prepare Fourier transformation arrays */
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(4);
  }

  /* initialize contact_term */
  contact_term[0] = 0.;
  contact_term[1] = 0.;
  contact_term[2] = 0.;
  contact_term[3] = 0.;
  contact_term[4] = 0.;
  contact_term[5] = 0.;
  contact_term[6] = 0.;
  contact_term[7] = 0.;

  /* determine source coordinates, find out, if source_location is in this process */
  /* The way the nu-dependence is implemented now only works if MPI is not used. --> 
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
    
  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;
  if(have_source_flag==1) { 
    fprintf(stdout, "local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location[0] = g_ipt[sx0m1][sx1][sx2][sx3];
    source_location[1] = g_ipt[sx0][sx1m1][sx2][sx3];
    source_location[2] = g_ipt[sx0][sx1][sx2m1][sx3];
    source_location[3] = g_ipt[sx0][sx1][sx2][sx3m1];
    source_location[4] = g_ipt[sx0][sx1][sx2][sx3];
    _cm_eq_cm_ti_co(Usource[0], &cvc_gauge_field[_GGI(source_location[0],0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &cvc_gauge_field[_GGI(source_location[1],1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &cvc_gauge_field[_GGI(source_location[2],2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &cvc_gauge_field[_GGI(source_location[3],3)], &co_phase_up[3]);
  }
#ifdef MPI
  MPI_Gather(&have_source_flag, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "status[%1d]=%d\n", mu,status[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; status[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] have_source_flag = %d\n", g_cart_id, have_source_flag);
#else
  have_source_flag = 0;
#endif

#ifdef MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  /* initialize the Gamma matrices */

  // gamma_5:
  gperm[4][0] = gamma_permutation[5][ 0] / 6;
  gperm[4][1] = gamma_permutation[5][ 6] / 6;
  gperm[4][2] = gamma_permutation[5][12] / 6;
  gperm[4][3] = gamma_permutation[5][18] / 6;
  gperm_sign[4][0] = gamma_sign[5][ 0];
  gperm_sign[4][1] = gamma_sign[5][ 6];
  gperm_sign[4][2] = gamma_sign[5][12];
  gperm_sign[4][3] = gamma_sign[5][18];
  // gamma_nu gamma_5
  for(nu=0;nu<4;nu++) {
    // permutation
    gperm[nu][0] = gamma_permutation[6+nu][ 0] / 6;
    gperm[nu][1] = gamma_permutation[6+nu][ 6] / 6;
    gperm[nu][2] = gamma_permutation[6+nu][12] / 6;
    gperm[nu][3] = gamma_permutation[6+nu][18] / 6;
    // is imaginary ?
    isimag[nu] = gamma_permutation[6+nu][0] % 2;
    // (overall) sign
    gperm_sign[nu][0] = gamma_sign[6+nu][ 0];
    gperm_sign[nu][1] = gamma_sign[6+nu][ 6];
    gperm_sign[nu][2] = gamma_sign[6+nu][12];
    gperm_sign[nu][3] = gamma_sign[6+nu][18];
    // write to stdout
    fprintf(stdout, "# gamma_%d5 = (%f %d, %f %d, %f %d, %f %d)\n", nu,
        gperm_sign[nu][0], gperm[nu][0], gperm_sign[nu][1], gperm[nu][1], 
        gperm_sign[nu][2], gperm[nu][2], gperm_sign[nu][3], gperm[nu][3]);
  }
  // gamma_nu
  for(nu=0;nu<4;nu++) {
    // permutation
    gperm2[nu][0] = gamma_permutation[nu][ 0] / 6;
    gperm2[nu][1] = gamma_permutation[nu][ 6] / 6;
    gperm2[nu][2] = gamma_permutation[nu][12] / 6;
    gperm2[nu][3] = gamma_permutation[nu][18] / 6;
    // (overall) sign
    gperm2_sign[nu][0] = gamma_sign[nu][ 0];
    gperm2_sign[nu][1] = gamma_sign[nu][ 6];
    gperm2_sign[nu][2] = gamma_sign[nu][12];
    gperm2_sign[nu][3] = gamma_sign[nu][18];
    // write to stdout
    fprintf(stdout, "# gamma_%d = (%f %d, %f %d, %f %d, %f %d)\n", nu,
        gperm2_sign[nu][0], gperm2[nu][0], gperm2_sign[nu][1], gperm2[nu][1], 
        gperm2_sign[nu][2], gperm2[nu][2], gperm2_sign[nu][3], gperm2[nu][3]);
  }

  /**********************************************************
   * read 12 up-type propagators with source source_location
   * - can get contributions 1 and 3 from that
   **********************************************************/
  for(ia=0; ia<12; ia++) {
    if(!mms) {
      get_filename(filename, 4, ia, 1); // position 4 is source position y, 1 means "+ i mu gamma_5", i.e. psource...
      read_lime_spinor(cvc_spinor_field[ia], filename, 0); //0 stands for first lime block
      xchange_field(cvc_spinor_field[ia]);
    } else {
      sprintf(filename, "%s.%.4d.04.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, ia, mass_id);
      read_lime_spinor(work, filename, 0);
      xchange_field(work);
      Qf5(cvc_spinor_field[ia], work, -g_mu);
      xchange_field(cvc_spinor_field[ia]);
    }
  }
      
/* to test gauge invariance [from avc_exact.c]*/  
/*    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, 4, ia, 1);
        read_lime_spinor(cvc_spinor_field[ia], filename, 0);
        xchange_field(cvc_spinor_field[ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop2(gauge_trafo, cvc_spinor_field[ia], ia/3, ia%3, 4, filename_prefix, g_source_location);
        xchange_field(cvc_spinor_field[ia]);
      }
    }*/



  /**********************************************
   * loop on the Lorentz index nu at source 
   **********************************************/
  for(nu=0; nu<4; nu++) {

    /* read 12 dn-type propagators */
    for(ia=0; ia<12; ia++) {
      if(!mms) {
        if(ud_one_file==0) {
          get_filename(filename, nu, ia, -1);//-1 implies reading from msource..
          read_lime_spinor(cvc_spinor_field[12+ia], filename, 0);
        } else {
          get_filename(filename, nu, ia, 1);
          read_lime_spinor(cvc_spinor_field[12+ia], filename, 1);
        }
        xchange_field(cvc_spinor_field[12+ia]);
      } else {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, nu, ia, mass_id);
        read_lime_spinor(work, filename, 0);
        xchange_field(work);
        Qf5(cvc_spinor_field[12+ia], work, g_mu);
        xchange_field(cvc_spinor_field[12+ia]);
      }
    }


/*to test gauge invariance [from avc_exact.c] */    
/*    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, nu, ia, -1);
        read_lime_spinor(cvc_spinor_field[12+ia], filename, 0);
        xchange_field(cvc_spinor_field[12+ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop2(gauge_trafo, cvc_spinor_field[12+ia], ia/3, ia%3, nu, filename_prefix2, g_source_location);
        xchange_field(cvc_spinor_field[12+ia]);
      }
    }*/
/**********************************************************************************************************/

   /* add new contractions to (existing) disc */
    for(ir=0; ir<4; ir++) { /*spinor index*/
      for(ia=0; ia<3; ia++) { /*colour index*/
        phi = cvc_spinor_field[3*ir+ia];
      for(ib=0; ib<3; ib++) { /*colour index */
        chi = cvc_spinor_field[12+3*gperm[nu][ir]+ib];
        fprintf(stdout, "\n# [nu5] spin index pair (%d, %d); col index pair (%d, %d)\n", ir, gperm[nu][ir], ia ,ib);

        // 1) gamma_nu gamma_5 x U^dagger
	// difference to cvc.c: now U^dagger instead of U
        for(mu=0; mu<4; mu++) {

          imunu = 4*mu+nu;

        /* first contribution */
#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif

      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */
           _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]); /* (from antip. bc in time direction) above only for source location ? */

	    _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
//            _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
	    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
            if(!isimag[nu]) {
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
            } else {
	      //TO BE CHECKED!!!
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im; 
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
            }
            //if(ix==0) fprintf(stdout, "[1_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[1_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[nu][ir] * w1.re, gperm_sign[nu][ir] * w1.im);

          }  // of ix


       /* second contribution */
#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
//            _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
	    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));	    
            if(!isimag[nu]) {
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
            } else {
	      	      //THIS HAS TO BE CHECKED!!
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
            }
            //if(ix==0) fprintf(stdout, "[3_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[3_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[nu][ir] * w1.re, gperm_sign[nu][ir] * w1.im);

          }  // of ix
	} // of mu
      } /* of ib */

      for(ib=0; ib<3; ib++) {
        chi = cvc_spinor_field[12+3*gperm[4][ir]+ib];
        //fprintf(stdout, "\n# [5] spin index pair (%d, %d); col index pair (%d, %d)\n", ir, gperm[4][ir], ia ,ib);

        // gamma_5 x U^dagger
        for(mu=0; mu<4; mu++) {
        //for(mu=0; mu<1; mu++) {
          imunu = 4*mu+nu;

      /* first contribution */
#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
//            _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
	    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
            conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
            conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
            //if(ix==0) fprintf(stdout, "[1_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[1_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[4][ir] * w1.re, gperm_sign[4][ir] * w1.im);

          }  // of ix

     /* second contribution */
#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
//            _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));  just in case this is needed to copy it later
	    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
            conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
            conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
            //if(ix==0) fprintf(stdout, "[3_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[3_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[4][ir] * w1.re, gperm_sign[4][ir] * w1.im);

          }  // of ix
	}  // of mu
      }  // of ib

      /* contribution to contact term (actually - contact_term)*/
      _fv_eq_cm_dag_ti_fv(spinor1, Usource[nu], phi+_GSI(g_idn[source_location[4]][nu]));
      _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
      _fv_pl_eq_fv(spinor2, spinor1);
      contact_term[2*nu  ] += 0.5 * spinor2[2*(3*ir+ia)  ];
      contact_term[2*nu+1] += 0.5 * spinor2[2*(3*ir+ia)+1];

      }  // of ia
    }  // of ir

  }  // of nu

  fprintf(stdout, "\n# [cvc] contact term after 1st part:\n");
  fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 0, contact_term[0], contact_term[1]);
  fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 1, contact_term[2], contact_term[3]);
  fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 2, contact_term[4], contact_term[5]);
  fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 3, contact_term[6], contact_term[7]);

  /**********************************************************
   * read 12 dn-type propagators with source source_location
   * - can get contributions 2 and 4 from that
   **********************************************************/
  for(ia=0; ia<12; ia++) {
    if(!mms) {
      if(ud_one_file==0) {
        get_filename(filename, 4, ia, -1);
        read_lime_spinor(cvc_spinor_field[12+ia], filename, 0);
      } else {
        get_filename(filename, 4, ia, 1);
        read_lime_spinor(cvc_spinor_field[12+ia], filename, 1);
      }
      xchange_field(cvc_spinor_field[12+ia]);
    } else {
      sprintf(filename, "%s.%.4d.04.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, ia, mass_id);
      read_lime_spinor(work, filename, 0);
      xchange_field(work);
      Qf5(cvc_spinor_field[12+ia], work, g_mu);
      xchange_field(cvc_spinor_field[12+ia]);
    }
  }
  

/*to test gauge invariance [from avc_exact.c] */
/*    if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, 4, ia, -1);
        read_lime_spinor(cvc_spinor_field[12+ia], filename, 0);
        xchange_field(cvc_spinor_field[12+ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop2(gauge_trafo, cvc_spinor_field[12+ia], ia/3, ia%3, 4, filename_prefix2, g_source_location);
        xchange_field(cvc_spinor_field[12+ia]);
      }
    }*/


/**********************************************
   * loop on the Lorentz index nu at source 
   **********************************************/
  for(nu=0; nu<4; nu++) {

    /* read 12 up-type propagators */
    for(ia=0; ia<12; ia++) {
      if(!mms) {
        get_filename(filename, nu, ia, 1);
        read_lime_spinor(cvc_spinor_field[ia], filename, 0);
        xchange_field(cvc_spinor_field[ia]);
      } else {
        sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, nu, ia, mass_id);
        read_lime_spinor(work, filename, 0);
        xchange_field(work);
        Qf5(cvc_spinor_field[ia], work, -g_mu);
        xchange_field(cvc_spinor_field[ia]);
      }
    }
    
/*to test gauge invariance [from avc_exact.c] */    
/*     if(do_gt==0) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, nu, ia, 1);
        read_lime_spinor(cvc_spinor_field[ia], filename, 0);
        xchange_field(cvc_spinor_field[ia]);
      }
    }
    else {
      for(ia=0; ia<12; ia++) {
        apply_gt_prop2(gauge_trafo, cvc_spinor_field[ia], ia/3, ia%3, nu, filename_prefix, g_source_location);
        xchange_field(cvc_spinor_field[ia]);
      }
    }*/
    
/*********************************************************************************************************************/

    for(ir=0; ir<4; ir++) {
      for(ia=0; ia<3; ia++) {
        phi = cvc_spinor_field[3*ir+ia];
      for(ib=0; ib<3; ib++) {
        chi = cvc_spinor_field[12+3*gperm[nu][ir]+ib];
        //fprintf(stdout, "\n# [nu5] spin index pair (%d, %d); col index pair (%d, %d)\n", ir, gperm[nu][ir], ia ,ib);
    
        // 1) gamma_nu gamma_5 x U
	// difference to cvc.c: now U instead of U^dagger
        for(mu=0; mu<4; mu++) {

          imunu = 4*mu+nu;

#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
//            _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
	    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
            if(!isimag[nu]) {
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
            } else {
	      // THIS HAS TO BE CHECKED!!
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
            }
            //if(ix==0) fprintf(stdout, "[2_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[2_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[nu][ir] * w1.re, gperm_sign[nu][ir] * w1.im);

          }  // of ix

#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
//            _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
	    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
            if(!isimag[nu]) {
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
            } else {
	      	      // THIS HAS TO BE CHECKED!!
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
            }
            //if(ix==0) fprintf(stdout, "[4_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[4_nu5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[nu][ir] * w1.re, gperm_sign[nu][ir] * w1.im);

          }  // of ix
	} // of mu

      } /* of ib */

      for(ib=0; ib<3; ib++) {
        chi = cvc_spinor_field[12+3*gperm[4][ir]+ib];
        //fprintf(stdout, "\n# [5] spin index pair (%d, %d); col index pair (%d, %d)\n", ir, gperm[4][ir], ia ,ib);

        // gamma_5 x U 
        for(mu=0; mu<4; mu++) {

          imunu = 4*mu+nu;

#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_mi_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
//            _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
	    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
            conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
            conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
            //if(ix==0) fprintf(stdout, "[2_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[2_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[4][ir] * w1.re, gperm_sign[4][ir] * w1.im);

          }  // of ix

#ifdef OPENMP
#pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu)
#endif
          for(ix=0; ix<VOLUME; ix++) {
            _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);

            _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
            _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	    _fv_pl_eq_fv(spinor2, spinor1);
	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
//            _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
            _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
            conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
            conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
            //if(ix==0) fprintf(stdout, "[4_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib, w.re, w.im);
            //if(ix==0) fprintf(stdout, "[4_5] %3d%3d%3d\t%e\t%e\n", ir ,ia, ib,
            //    gperm_sign[4][ir] * w1.re, gperm_sign[4][ir] * w1.im);

          }  // of ix
	}  // of mu
      }  // of ib

      /* contribution to contact term */
      // TO BE CHECKED.
      _fv_eq_cm_ti_fv(spinor1, Usource[nu], phi+_GSI(source_location[4]));
      _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
      _fv_mi_eq_fv(spinor2, spinor1);
      contact_term[2*nu  ] += -0.5 * spinor2[2*(3*ir+ia)  ];
      contact_term[2*nu+1] += -0.5 * spinor2[2*(3*ir+ia)+1];

      }  // of ia 
    }  // of ir
  }  // of nu
  
  // print contact term
  if(g_cart_id==0) {
    fprintf(stdout, "\n# [cvc] contact term\n");
    for(i=0;i<4;i++) {
      fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
    }
  }

  /* normalisation of contractions */
#ifdef OPENMP
#pragma omp parallel for
#endif
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] *= -0.25;

#ifdef MPI
      retime = MPI_Wtime();
#else
      retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "contractions in %e seconds\n", retime-ratime);

  /* save results in position space */
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(outfile_prefix_set) {
    sprintf(filename, "%s/cvc_v_x.%.4d", outfile_prefix, Nconf);
  } else {
    sprintf(filename, "cvc_v_x.%.4d", Nconf);
  }
  sprintf(contype, "cvc - cvc in position space, all 16 components");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);
  if(write_ascii) {
    if(outfile_prefix_set) {
      sprintf(filename, "%s/cvc_v_x.%.4di.ascii", outfile_prefix, Nconf);
    } else {
      sprintf(filename, "cvc_v_x.%.4d.ascii", Nconf);
    }
    write_contraction(conn, NULL, filename, 16, 2, 0);
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);

#ifndef MPI
  /* check the Ward identity in position space */
  if(check_position_space_WI) {
    sprintf(filename, "WI_X.%.4d", Nconf);
    ofs = fopen(filename,"w");
    fprintf(stdout, "\n# [cvc] checking Ward identity in position space ...\n");
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
      ix=g_ipt[x0][x1][x2][x3];
      for(nu=0; nu<4; nu++) {
        w.re = conn[_GWI(4*0+nu,ix,VOLUME)] + conn[_GWI(4*1+nu,ix,VOLUME)]
             + conn[_GWI(4*2+nu,ix,VOLUME)] + conn[_GWI(4*3+nu,ix,VOLUME)]
	     - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	     - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];

        w.im = conn[_GWI(4*0+nu,ix,VOLUME)+1] + conn[_GWI(4*1+nu,ix,VOLUME)+1]
            + conn[_GWI(4*2+nu,ix,VOLUME)+1] + conn[_GWI(4*3+nu,ix,VOLUME)+1]
	    - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
	    - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
        fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, w.re, w.im);
      }
    }}}}
    fclose(ofs);
  }
#endif

  /*********************************************
   * Fourier transformation 
   *********************************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
#  ifdef OPENMP
    fftwnd_threads_one(num_threads, plan_p, in, NULL);
#  else
    fftwnd_one(plan_p, in, NULL);
#  endif
#endif
    memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

#ifdef MPI
  if(g_cart_id==0) fprintf(stdout, "\n# [] broadcasing contact term ...\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact term = "\
      "(%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI,%12.5e+%12.5eI)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]);
#endif

  /*****************************************
   * add phase factors
   * To be checked.
   *****************************************/
  for(mu=0; mu<4; mu++) {
    phi = conn + _GWI(5*mu,0,VOLUME);

    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * (double)(Tstart+x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re = cos(phase[mu]- (phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im = sin(phase[mu]- (phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      _co_eq_co_ti_co(&w1,(complex*)( phi+2*ix ), &w);
      phi[2*ix  ] = w1.re - contact_term[2*mu  ];
      phi[2*ix+1] = w1.im - contact_term[2*mu+1];
    }}}}
  }  /* of mu */

//test: leave out phase factor
  for(mu=0; mu<3; mu++) {
    for(nu=mu+1; nu<4; nu++) {
     phi = conn + _GWI(4*mu+nu,0,VOLUME);
     chi = conn + _GWI(4*nu+mu,0,VOLUME);

     for(x0=0; x0<T; x0++) {
       phase[0] =  (double)(Tstart+x0) * M_PI / (double)T_global;
     for(x1=0; x1<LX; x1++) {
       phase[1] =  (double)(x1) * M_PI / (double)LX;
     for(x2=0; x2<LY; x2++) {
       phase[2] =  (double)(x2) * M_PI / (double)LY;
     for(x3=0; x3<LZ; x3++) {
       phase[3] =  (double)(x3) * M_PI / (double)LZ;
       ix = g_ipt[x0][x1][x2][x3];
       w.re =  cos( phase[mu] + phase[nu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
       w.im =  sin( phase[mu] + phase[nu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
       _co_eq_co_ti_co(&w1,(complex*)( phi+2*ix ), &w);
       phi[2*ix  ] = w1.re;
       phi[2*ix+1] = w1.im;

       w.re =  cos( phase[nu] + phase[mu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
       w.im =  sin( phase[nu] + phase[mu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
       _co_eq_co_ti_co(&w1,(complex*)( chi+2*ix ), &w);
       chi[2*ix  ] = w1.re;
       chi[2*ix+1] = w1.im;
    }}}}
  }}  /* of mu and nu */

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);

  /********************************
   * save momentum space results
   ********************************/
#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(outfile_prefix_set) {
    sprintf(filename, "%s/cvc_v_p.%.4d", outfile_prefix, Nconf);
  } else {
    sprintf(filename, "cvc_v_p.%.4d", Nconf);
  }
  sprintf(contype, "cvc - cvc in momentum space, all 16 components");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);
  if(write_ascii) {
    if(outfile_prefix_set) {
      sprintf(filename, "%s/cvc_v_p.%.4d", outfile_prefix, Nconf);
    } else {
      sprintf(filename, "cvc_v_p.%.4d.ascii", Nconf);
    }
    write_contraction(conn, (int*)NULL, filename, 16, 2, 0);
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved momentum space results in %e seconds\n", retime-ratime);

  if(check_momentum_space_WI) {
    sprintf(filename, "WI_P.%.4d", Nconf);
    ofs = fopen(filename,"w");
    fprintf(stdout, "\n# [cvc] checking Ward identity in momentum space ...\n");
    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * sin( (double)(Tstart+x0) * M_PI / (double)T_global );
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * sin( (double)(x1) * M_PI / (double)LX );
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * sin( (double)(x2) * M_PI / (double)LY );
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * sin( (double)(x3) * M_PI / (double)LZ );
      ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
      for(nu=0;nu<4;nu++) {
        w.re = phase[0] * conn[_GWI(4*0+nu,ix,VOLUME)] + phase[1] * conn[_GWI(4*1+nu,ix,VOLUME)] 
             + phase[2] * conn[_GWI(4*2+nu,ix,VOLUME)] + phase[3] * conn[_GWI(4*3+nu,ix,VOLUME)];

        w.im = phase[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] + phase[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] 
             + phase[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] + phase[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];

        w1.re = phase[0] * conn[_GWI(4*nu+0,ix,VOLUME)] + phase[1] * conn[_GWI(4*nu+1,ix,VOLUME)] 
              + phase[2] * conn[_GWI(4*nu+2,ix,VOLUME)] + phase[3] * conn[_GWI(4*nu+3,ix,VOLUME)];

        w1.im = phase[0] * conn[_GWI(4*nu+0,ix,VOLUME)+1] + phase[1] * conn[_GWI(4*nu+1,ix,VOLUME)+1] 
              + phase[2] * conn[_GWI(4*nu+2,ix,VOLUME)+1] + phase[3] * conn[_GWI(4*nu+3,ix,VOLUME)+1];
        fprintf(ofs, "\t%d%25.16e%25.16e%25.16e%25.16e\n", nu, w.re, w.im, w1.re, w1.im);
      }
    }}}}
    fclose(ofs);
  }

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  free(cvc_gauge_field);
  for(i=0; i<no_fields; i++) free(cvc_spinor_field[i]);
  free(cvc_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif

  g_the_time = time(NULL);
  fprintf(stdout, "\n# [cvc] %s# [cvc] end of run\n", ctime(&g_the_time));
  fprintf(stderr, "\n# [cvc] %s# [cvc] end of run\n", ctime(&g_the_time));

  return(0);

}

