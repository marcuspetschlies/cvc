/******************************************************
* cvc_Vsources.c
* July, 4th, 2011
******************************************************/
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
  fprintf(stdout, "Code to perform contractions for conn. contrib. to vac. pol. from propagators obtained with volume sources\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -a write data also in ascii format and not only in binary lime format\n");
  fprintf(stdout, "         -W check Ward Identity in momentum space\n");
#ifdef MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

int main(int argc, char **argv) {

  int c, i, mu, nu;
  int count        = 0;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iy;
  int sid, sid2;
  double *conn  = (double*)NULL;
  double *work = (double*)NULL;
//  double *xspace = (double*)NULL;
  double q[4], fnorm;
  double unit_trace[2], shift_trace[2], D_trace[2];
  int verbose = 0;
  int do_gt   = 0;
  int check_momentum_space_WI=0;
  int write_ascii=0;
  char filename[100],contype[400];
  double ratime, retime;
  double plaq;
  double phase[4];
  double spinor1[24], spinor2[24], U_[18];
  double contact_term[8], buffer[8];
  double *gauge_trafo=(double*)NULL;
  complex w, w1, *cp1, *cp2, *cp3;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef MPI
  fftwnd_mpi_plan plan_p, plan_m;
  int *status;
#else
  fftwnd_plan plan_p, plan_m;
#endif

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  /* read the option arguments */
  while ((c = getopt(argc, argv, "Wah?vgf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'g':
      do_gt = 1;
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf(stdout, "\n# [] will check Ward identity in momentum space\n");
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [] will write data in ASCII format too\n");
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
  set_default_input_values();
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);

  /* read the input file */
//  read_input(filename);
  read_input_parser(filename);
  printf("Nconf=%d, g_sourceid= %d, g_sourceid2=%d\n", Nconf, g_sourceid, g_sourceid2);

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
  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  plan_m = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_FORWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  plan_m = fftwnd_create_plan(4, dims, FFTW_FORWARD,  FFTW_MEASURE | FFTW_IN_PLACE);
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

  /* read the gauge field */
/*  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
  read_lime_gauge_field_doubleprec(filename);*/

#ifdef MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "measured plaquette value: %25.16e\n", plaq);

   /* gauge transformation */
   if(do_gt==1) {
    init_gauge_trafo(&gauge_trafo, 1.);
    apply_gt_gauge(gauge_trafo);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "measured plaquette value after gauge trafo: %25.16e\n", plaq);
  }

  /* allocate memory for the spinor fields */
  no_fields = 4; 
  cvc_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&cvc_spinor_field[i], VOLUMEPLUSRAND);

  /* allocate memory for the contractions */
  conn = (double*)calloc(16*VOLUME, sizeof(double)); //4 for mu, 4 for nu
  if( conn == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for conn\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }

#ifdef OPENMP
#pragma omp parallel for
#endif

  work  = (double*)calloc(32*VOLUME, sizeof(double));
  if( work == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for work\n");
    #  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }
    
//  xspace  = (double*)calloc(32*VOLUME, sizeof(double));/* 4 for mu, 4 for nu, volume for x, volume for y ??*/
/*  if( xspace == (double*)NULL ) { 
    fprintf(stderr, "could not allocate memory for xspace\n");
#  ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#  endif
    exit(3);
  }*/

  /* initialise connected contribution to vacuum polarisation */
  for(ix=0; ix<16*VOLUME; ix++) conn[ix] = 0.;

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

  /***********************************************
   * start loop on source id 
   ***********************************************/
  for(sid=g_sourceid; sid<g_sourceid2; sid++) {

    /* read the new propagator for sid */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid);
      if(read_lime_spinor(cvc_spinor_field[1], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid);
      if(read_cmi(cvc_spinor_field[1], filename) != 0) break;
    }
    #ifdef MPI
    xchange_field(cvc_spinor_field[1]);
    #endif

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to read prop. for sid = %d: %e seconds\n", sid, retime-ratime);


   /*  gauge transform the propagators for sid */
    if(do_gt==1) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_cm_ti_fv(spinor1, gauge_trafo+18*ix, cvc_spinor_field[1]+_GSI(ix));
        _fv_eq_fv(cvc_spinor_field[1]+_GSI(ix), spinor1);
      }
    #ifdef MPI
      xchange_field(cvc_spinor_field[1]);
    #endif
    }

    /* calculate the source for sid: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    Q_phi_tbc(cvc_spinor_field[0], cvc_spinor_field[1]);

#ifdef MPI
    xchange_field(cvc_spinor_field[0]); 
#endif

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to calculate source for sid=%d: %e seconds\n", sid, retime-ratime);

  /******************************
   * second loop on source id
   ******************************/
  for(sid2=sid+1; sid2<=g_sourceid2; sid2++) {

    /* read the new propagator for sid2 */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    if(format==0) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sid2);
      if(read_lime_spinor(cvc_spinor_field[3], filename, 0) != 0) break;
    }
    else if(format==1) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, sid2);
      if(read_cmi(cvc_spinor_field[3], filename) != 0) break;
    }

    #ifdef MPI
    xchange_field(cvc_spinor_field[3]);
    #endif

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    fprintf(stdout, "time to read prop. for sid2=%d: %e seconds\n", sid2, retime-ratime);


    /* gauge transform the propagators for sid2 */
    if(do_gt==1) {
      for(ix=0; ix<VOLUME; ix++) {
        _fv_eq_cm_ti_fv(spinor1, gauge_trafo+18*ix, cvc_spinor_field[3]+_GSI(ix));
        _fv_eq_fv(cvc_spinor_field[3]+_GSI(ix), spinor1);
      }
    #ifdef MPI
      xchange_field(cvc_spinor_field[3]);
    #endif
    }

    /* calculate the source: apply Q_phi_tbc */
#ifdef MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    Q_phi_tbc(cvc_spinor_field[2], cvc_spinor_field[3]);

    xchange_field(cvc_spinor_field[2]);

#ifdef MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "time to calculate source for sid2=%d: %e seconds\n", sid2, retime-ratime);

   count++;

    /* add new contractions to (existing) disc */
#  ifdef MPI
    ratime = MPI_Wtime();
#  else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#  endif

    for(mu=0; mu<4; mu++) { /* loop on Lorentz index of the current */
      iix = _GWI(mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */
        _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]);

        /* first contribution */
        _fv_eq_cm_ti_fv(spinor1, U_, &cvc_spinor_field[1][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_mi_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[2][_GSI(ix)], spinor2);
	conn[iix  ] = -0.5 * w.re;
	conn[iix+1] = -0.5 * w.im;


        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &cvc_spinor_field[1][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[2][_GSI(g_iup[ix][mu])], spinor2);
	conn[iix  ] -= 0.5 * w.re;
	conn[iix+1] -= 0.5 * w.im;

	iix += 2;

      }  /* of ix */
    }    /* of mu */

    for(mu=0; mu<4; mu++) { /* loop on Lorentz index of the current */
      iix = _GWI(4+mu,0,VOLUME);
      for(ix=0; ix<VOLUME; ix++) {    /* loop on lattice sites */
        _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]);

        /* first contribution */
        _fv_eq_cm_ti_fv(spinor1, U_, &cvc_spinor_field[3][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_mi_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[0][_GSI(ix)], spinor2);
	conn[iix  ] = -0.5 * w.re;
	conn[iix+1] = -0.5 * w.im;

        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &cvc_spinor_field[3][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	conn[iix  ] -= 0.5 * w.re;
	conn[iix+1] -= 0.5 * w.im;

	iix += 2;

      }  /* of ix */
    }    /* of mu */

#  ifdef MPI
    retime = MPI_Wtime();
#  else
    retime = (double)clock() / CLOCKS_PER_SEC;
#  endif
    fprintf(stdout, "[%2d] contractions for sid pair (%d,%d) in %e seconds\n", g_cart_id, sid, sid2, retime-ratime);
    
    
    /* Try to produce results in position space to be saved later. -> question: avoid equal points??, xspace also dependent on iy??*/
    /* add contrib. to xspace */
/*    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      for(ix=0; ix<VOLUME; ix++) {
	for(iy=0; abs(ix-iy)<6; iy++){
        _co_eq_co_ti_co(&w, (complex*)(conn+_GWI(mu,ix,VOLUME)), (complex*)(conn+_GWI(4+nu,iy,VOLUME)));
        xspace[_GWI(4*mu+nu,ix,VOLUME)  ] += w.re;
        xspace[_GWI(4*mu+nu,ix,VOLUME)+1] += w.im;
	}
      }
    }
    }*/
    
    

    /*  Fourier transform data, add to work */
/*#ifdef MPI                              Since there is no retime, this ratime seems not to be needed.
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif */

    for(mu=0; mu<4; mu++) {
      memcpy((void*)in, (void*)(conn+_GWI(mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_p, in, NULL);
#endif
      memcpy((void*)(conn+_GWI(mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));

      memcpy((void*)in, (void*)(conn+_GWI(4+mu,0,VOLUME)), 2*VOLUME*sizeof(double));
#ifdef MPI
      fftwnd_mpi(plan_m, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
      fftwnd_one(plan_m, in, NULL);
#endif
      memcpy((void*)(conn+_GWI(4+mu,0,VOLUME)), (void*)in, 2*VOLUME*sizeof(double));
    }  /* of mu =0 ,..., 3*/

    /* add contrib. to work */
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      for(ix=0; ix<VOLUME; ix++) {
        _co_eq_co_ti_co(&w, (complex*)(conn+_GWI(mu,ix,VOLUME)), (complex*)(conn+_GWI(4+nu,ix,VOLUME)));
        work[_GWI(4*mu+nu,ix,VOLUME)  ] += w.re;
        work[_GWI(4*mu+nu,ix,VOLUME)+1] += w.im;
      }
    }
    }

   


    if(g_cart_id==0) fprintf(stdout, "-------------------------------------------------------\n");

  }  /* of sid2 */

    /*  add contrib. to contact term */
    for(mu=0; mu<4; mu++) {
      for(ix=0; ix<VOLUME; ix++) { 
        _cm_eq_cm_ti_co(U_, &cvc_gauge_field[_GGI(ix, mu)], &co_phase_up[mu]);

        /* first contribution */
        _fv_eq_cm_ti_fv(spinor1, U_, &cvc_spinor_field[1][_GSI(g_iup[ix][mu])]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_mi_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[0][_GSI(ix)], spinor2);
	contact_term[2*mu  ] += 0.5 * w.re;
	contact_term[2*mu+1] += 0.5 * w.im;


        /* second contribution */
	_fv_eq_cm_dag_ti_fv(spinor1, U_, &cvc_spinor_field[1][_GSI(ix)]);
	_fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
	_fv_pl_eq_fv(spinor2, spinor1);
	_co_eq_fv_dag_ti_fv(&w, &cvc_spinor_field[0][_GSI(g_iup[ix][mu])], spinor2);
	contact_term[2*mu  ] -= 0.5 * w.re; 
	contact_term[2*mu+1] -= 0.5 * w.im;
      }
    }

  }  /* of sid  */

  /***********************************
   * calculate final contact term
   ***********************************/
#ifdef MPI
  for(mu=0; mu<8; mu++) buffer[mu] = contact_term[mu];
  MPI_Allreduce(buffer, contact_term, 8, MPI_DOUBLE, MPI_SUM, g_cart_grid);
#endif
// add 1/2 due to normalisation of sources (1/sqrt(2)) for each spinor
  fnorm = 1. / ( (double)(T_global*LX*LY*LZ) * (double)(g_sourceid2-g_sourceid) * 2. );
  fprintf(stdout, "contact term fnorm = %25.16e\n", fnorm);
  for(mu=0; mu<8; mu++) contact_term[mu] *= fnorm;
  if(g_cart_id==0) {
    fprintf(stdout, "contact term:\n");
    for(mu=0; mu<4; mu++) fprintf(stdout, "%3d%25.16e%25.16e\n", mu, contact_term[2*mu], contact_term[2*mu+1]);
  }

  /*********************************************************
   * add phase factor, normalize and subtract contact term
   * - note minus sign in fnorm from fermion loop
   *********************************************************/
// add 1/4 due to normalisation of sources (1/sqrt(2)) for each spinor because all result from sources not normalised in invert code
  fnorm = -1. / ( (double)(T_global*LX*LY*LZ) * (double)(count) *4. );
  fprintf(stdout, "Pi fnorm = %e\n", fnorm);
  
  /*position space */
/*  for(mu=0; mu<4; mu++) {
  for(nu=0; nu<4; nu++) {
  _co_eq_co_ti_re((complex*)(xspace+_GWI(4*mu+nu,ix,VOLUME)),(complex*)(xspace+_GWI(4*mu+nu,ix,VOLUME)) , fnorm);*/
/*  if(mu == nu) {              in cvc.c xspace results without contact term
        xspace[_GWI(4*mu+nu,ix,VOLUME)  ] += contact_term[2*mu  ];
        xspace[_GWI(4*mu+nu,ix,VOLUME)+1] += contact_term[2*mu+1];
      } */
//  }
//  }
  
    /* save results in position space */
/*#ifdef MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  sprintf(filename, "cvc_v_x.%.4d", Nconf);
  sprintf(contype, "cvc - cvc in position space, all 16 components");
  write_lime_contraction(xspace, filename, 64, 16, contype, Nconf, 0);
  if(write_ascii) {
    sprintf(filename, "cvc_v_x.%.4d.ascii", Nconf);
    write_contraction(xspace, NULL, filename, 16, 2, 0);
  }

#ifdef MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "saved position space results in %e seconds\n", retime-ratime);*/
  
  
  /* momentum space */
  // Tried version from cvc.c -> same result. 
  for(mu=0; mu<4; mu++) {
  for(nu=0; nu<4; nu++) {
     
    for(x0=0; x0<T; x0++) {
      q[0] = (double)(x0+Tstart) / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] = (double)(x1) / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = (double)(x2) / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = (double)(x3) / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re = cos( M_PI * (q[mu]-q[nu]) );
      w.im = sin( M_PI * (q[mu]-q[nu]) );
      _co_eq_co_ti_co(&w1, (complex*)(work+_GWI(4*mu+nu,ix,VOLUME)), &w);
      _co_eq_co_ti_re((complex*)(work+_GWI(4*mu+nu,ix,VOLUME)), &w1, fnorm);
      //check convergence of results with increasing number of sources without adding contact term
      if(mu == nu) {
        work[_GWI(4*mu+nu,ix,VOLUME)  ] += contact_term[2*mu  ];
        work[_GWI(4*mu+nu,ix,VOLUME)+1] += contact_term[2*mu+1];
      }
      //test whether lattice artefacts are responsible for difference between volume and point sources
//      work[_GWI(4*mu+nu,ix,VOLUME)+1] =0; -> zu spät
    }
    }
    }
    }

  }
  }

/*  for(mu=0; mu<4; mu++) {
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      _co_eq_co_ti_re((complex*)(work+_GWI(5*mu,ix,VOLUME)),(complex*)(work+_GWI(5*mu,ix,VOLUME)) , fnorm);
      work[_GWI(5*mu,ix,VOLUME) ]  += contact_term[2*mu  ];
      work[_GWI(5*mu,ix,VOLUME)+1] += contact_term[2*mu+1];
    }}}}
  }*/  /* of mu */

/*  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {

    for(x0=0; x0<T; x0++) {
      q[0] =  (double)(Tstart+x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      q[1] =  (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] =  (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] =  (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re =  cos( q[mu] - q[nu]);
      w.im =  sin( q[mu] - q[nu]);
      _co_eq_co_ti_co(&w1,(complex*)(work+_GWI(4*mu+nu,ix,VOLUME)  ), &w);
      _co_eq_co_ti_re((complex*)(work+_GWI(4*mu+nu,ix,VOLUME)), &w1, fnorm);
    
      w.re =  cos( q[nu] - q[mu]);
      w.im =  sin( q[nu] - q[mu]);
      _co_eq_co_ti_co(&w1,(complex*)(work+_GWI(4*nu+mu,ix,VOLUME)), &w);
      _co_eq_co_ti_re((complex*)(work+_GWI(4*nu+mu,ix,VOLUME)), &w1, fnorm);
    }}}}
  }}*/  /* of mu and nu */



  /*****************************************
   * save the result in momentum space
   *****************************************/
//  sprintf(filename, "cvc_v_p.%.4d.%.4d_%.4d", Nconf, g_sourceid, g_sourceid2);
  sprintf(filename, "cvc_v_p.%.4d", Nconf);
  sprintf(contype, "cvc - cvc in position space, all 16 components");
  write_lime_contraction(work, filename, 64, 16, contype, Nconf, 0);
  
  if(write_ascii) {
    sprintf(filename, "cvc_v_p.%.4d.ascii", Nconf);
    write_contraction(work,(int*) NULL, filename, 16, 2, 0);
  }

  fprintf(stdout, "Results in momentum space have been saved.\n");
  
  /********************************************
  * check the WI in momentum space
  ********************************************/
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
        w.re = phase[0] * work[_GWI(4*0+nu,ix,VOLUME)] + phase[1] * work[_GWI(4*1+nu,ix,VOLUME)] 
             + phase[2] * work[_GWI(4*2+nu,ix,VOLUME)] + phase[3] * work[_GWI(4*3+nu,ix,VOLUME)];

        w.im = phase[0] * work[_GWI(4*0+nu,ix,VOLUME)+1] + phase[1] * work[_GWI(4*1+nu,ix,VOLUME)+1] 
             + phase[2] * work[_GWI(4*2+nu,ix,VOLUME)+1] + phase[3] * work[_GWI(4*3+nu,ix,VOLUME)+1];

        w1.re = phase[0] * work[_GWI(4*nu+0,ix,VOLUME)] + phase[1] * work[_GWI(4*nu+1,ix,VOLUME)] 
              + phase[2] * work[_GWI(4*nu+2,ix,VOLUME)] + phase[3] * work[_GWI(4*nu+3,ix,VOLUME)];

        w1.im = phase[0] * work[_GWI(4*nu+0,ix,VOLUME)+1] + phase[1] * work[_GWI(4*nu+1,ix,VOLUME)+1] 
              + phase[2] * work[_GWI(4*nu+2,ix,VOLUME)+1] + phase[3] * work[_GWI(4*nu+3,ix,VOLUME)+1];
        fprintf(ofs, "\t%d%25.16e%25.16e%25.16e%25.16e\n", nu, w.re, w.im, w1.re, w1.im);
      }
    }}}}
    fclose(ofs);
  }


  /*****************************************
   * free the allocated memory, finalize
   *****************************************/
  free(cvc_gauge_field);
  for(i=0; i<no_fields; i++) free(cvc_spinor_field[i]);
  free(cvc_spinor_field);
  free_geometry();
  fftw_free(in);
  free(conn);
  free(work);
//  free(xspace);
  if(do_gt==1) free(gauge_trafo);

#ifdef MPI
  fftwnd_mpi_destroy_plan(plan_p);
  fftwnd_mpi_destroy_plan(plan_m);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
  fftwnd_destroy_plan(plan_m);
#endif

  return(0);

}