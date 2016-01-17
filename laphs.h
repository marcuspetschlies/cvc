#ifndef _LAPHS_H
#define _LAPHS_H

#if defined _LAPHS_UTILS
#  define LAPHS_EXTERN
#else
#  define LAPHS_EXTERN extern
#endif 

namespace cvc {


LAPHS_EXTERN char laphs_time_proj_type[200];
LAPHS_EXTERN char laphs_spin_proj_type[200];
LAPHS_EXTERN char laphs_evec_proj_type[200];

LAPHS_EXTERN int laphs_eigenvector_number;
LAPHS_EXTERN int laphs_randomvector_number;

LAPHS_EXTERN char laphs_eigenvector_path_prefix[200];
LAPHS_EXTERN char laphs_randomvector_path_prefix[200];
LAPHS_EXTERN char laphs_perambulator_path_prefix[200];

LAPHS_EXTERN char laphs_eigenvector_file_prefix[200];
LAPHS_EXTERN char laphs_eigenvalue_file_prefix[200];
LAPHS_EXTERN char laphs_phase_file_prefix[200];

LAPHS_EXTERN char laphs_perambulator_file_prefix[200];
LAPHS_EXTERN char laphs_randomvector_file_prefix[200];

LAPHS_EXTERN int laphs_spin_src_number;
LAPHS_EXTERN int laphs_evec_src_number;
LAPHS_EXTERN int laphs_time_src_number;

#define _default_perambulator_quark_type "x"
#define _default_perambulator_snk_type  "NA"

typedef struct {
  int nt, nv;
  double ***v;
  double *evec;
  double *eval;
  double *phase;
} eigensystem_type;



typedef struct {
  int nt;
  int ns;
  int nv;
  double *rvec;
} randomvector_type;


typedef struct {
  int nt_src, ns_src, nv_src;
  int nt_snk, ns_snk, nv_snk;
  double ****v;
  double *p;
  char quark_type[2];
  char snk_type[20];
  int irnd;
  int nc_snk;
} perambulator_type;

}

#endif
