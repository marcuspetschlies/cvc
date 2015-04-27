#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#if defined ( MPI )
#  include <mpi.h>
#endif

//#include "ifftw.h"

#if defined MAIN_PROGRAM
#  define EXTERN
#else
#  define EXTERN extern
#endif 

#include "cvc_complex.h"

#define _GSI(_ix) (24*(_ix))
#define _GGI(_ix,_mu) (18*(4*(_ix)+(_mu)))
#define _GJI(_ix,_mu) (2*(4*(_ix)+(_mu)))
#define _GWI(_ix,_mu,_N) (2*((_N)*(_ix)+(_mu)))

#ifndef _Q2EPS
#  define _Q2EPS (5.e-14)
#endif

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#define HPE_MAX_ORDER 11

#define _MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b) )



#ifdef LIB_WRAPPER

  extern int T_global, LX_global, LY_global;
  extern int T, L, LX, LY, LZ, VOLUME, ;
  extern int RAND, EDGES, VOLUMEPLUSRAND;

  extern int **** g_ipt;
  extern int ** g_iup;
  extern int ** g_idn;

  extern int g_proc_id, g_nproc;
  extern int g_cart_id;
  extern int g_nb_list[8];
  extern int g_proc_coords[4];

  #ifdef MPI
  extern MPI_Comm g_cart_grid;
  extern MPI_Status status;
  extern MPI_Comm g_ts_comm, g_xs_comm;
  #endif

  extern int g_nb_t_up, g_nb_t_dn;
  extern int g_nb_x_up, g_nb_x_dn;
  extern int g_nb_y_up, g_nb_y_dn;
  extern int g_nb_z_up, g_nb_z_dn;
  extern int g_nproc_t, g_nproc_x, g_nproc_y, g_nproc_z;
  
  
#else //LIB_WRAPPER

  EXTERN int T_global, LX_global, LY_global;
  EXTERN int T, L, LX, LY, LZ, VOLUME;
  EXTERN int RAND, EDGES, VOLUMEPLUSRAND;

  EXTERN int **** g_ipt;
  EXTERN int ** g_iup;
  EXTERN int ** g_idn;

  EXTERN int g_proc_id, g_nproc;
  EXTERN int g_cart_id;
  EXTERN int g_nb_list[8];
  EXTERN int g_proc_coords[4];

  #ifdef MPI
  EXTERN MPI_Comm g_cart_grid;
  EXTERN MPI_Status status;
  EXTERN MPI_Comm g_ts_comm, g_xs_comm;
  #endif

  EXTERN int g_nb_t_up, g_nb_t_dn;
  EXTERN int g_nb_x_up, g_nb_x_dn;
  EXTERN int g_nb_y_up, g_nb_y_dn;
  EXTERN int g_nb_z_up, g_nb_z_dn;
  EXTERN int g_nproc_t, g_nproc_x, g_nproc_y, g_nproc_z;

  
#endif //LIB_WRAPPER






EXTERN int Nconf;

EXTERN int g_ts_id, g_ts_nproc;
EXTERN int g_xs_id, g_xs_nproc;
EXTERN int g_ts_nb_up, g_ts_nb_dn;
EXTERN int g_ts_nb_x_up, g_ts_nb_x_dn;
EXTERN int g_ts_nb_y_up, g_ts_nb_y_dn;



EXTERN double **cvc_spinor_field;
EXTERN double *cvc_gauge_field;
EXTERN double g_kappa, g_mu, g_musigma, g_mudelta;


EXTERN int FFTW_LOC_VOLUME, Tstart, LXstart, LYstart;




EXTERN int g_sourceid, g_sourceid2, g_sourceid_step, Nsave;
EXTERN int g_gaugeid, g_gaugeid2, g_gauge_step;

EXTERN char filename_prefix[200], filename_prefix2[200], gaugefilename_prefix[200];
EXTERN int format, rotate;
EXTERN double BCangle[4];

EXTERN int no_fields;

EXTERN complex co_phase_up[4];

EXTERN int gamma_permutation[16][24], gamma_sign[16][24];
EXTERN int perm_tab_3[6][3], perm_tab_4[24][4], perm_tab_3e[3][3], perm_tab_3o[3][3], perm_tab_4e[12][4], perm_tab_4o[12][4];

EXTERN int g_resume, g_subtract;
EXTERN int g_source_location;

EXTERN unsigned int g_seed;
EXTERN int g_noise_type, g_source_type;

EXTERN double solver_precision;
EXTERN int niter_max;
EXTERN int hpe_order_min, hpe_order_max, hpe_order;

EXTERN double g_cutangle, g_cutradius, g_rmin, g_rmax;
EXTERN int g_cutdir[4];

EXTERN int avgL, avgT;

EXTERN double model_dcoeff_re, model_dcoeff_im, model_mrho;
EXTERN double g_prop_normsqr;
EXTERN double g_qhatsqr_min, g_qhatsqr_max;

EXTERN int Nlong, N_ape, N_Jacobi;
EXTERN double alpha_ape, kappa_Jacobi;
EXTERN int g_source_timeslice, g_no_extra_masses, g_no_light_masses, g_no_strange_masses;

EXTERN int g_local_local, g_local_smeared, g_smeared_local, g_smeared_smeared;
EXTERN int g_rotate_ETMC_UKQCD;
EXTERN time_t g_the_time;
EXTERN int g_propagator_position;
#endif
