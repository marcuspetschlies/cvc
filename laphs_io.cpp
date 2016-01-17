#include <stdlib.h>
#include <stdio.h>
#include "global.h"
#include "laphs.h" 
#include "laphs_io.h" 
 
namespace cvc {


/************************************************************************************
 * read perambulator field
 *
 * tag is like "TsoB0024.VsoI0006.DsoF4.TsiF0048.SsiF13824.DsiF4.CsiF3.smeared1"
 ************************************************************************************/
int read_perambulator(perambulator_type *peram) {

  int col_i, row_i;
  int t1,       /* sink time */
      t2,       /* source time */
      ev1, ev2, /* eigenvector number */
      dirac1,   /* dirac index at sink */
      dirac2;   /* dirac index at source */
  int evec_src_number, evec_snk_number, spin_src_number, spin_snk_number, time_src_number, time_snk_number;
  int i1, i2, k1, k2;
  size_t ncol, nrow;
  size_t items, count;
  unsigned int idx;
  FILE *ifs;
  char filename[400], tag[400];
  double *buffer=NULL;
  unsigned int VOL3 = LX*LY*LZ;

  time_snk_number = peram->nt_snk;
  time_src_number = peram->nt_src;
  spin_snk_number = peram->ns_snk;
  spin_src_number = peram->ns_src;
  evec_snk_number = peram->nv_snk;
  evec_src_number = peram->nv_src;

  if( peram->p == NULL) {
    fprintf(stderr, "[read_perambulator] Error, perambulator field is NULL\n");
    return(1);
  }

  nrow = (size_t)( evec_snk_number * spin_snk_number * time_snk_number );
  ncol = (size_t)( evec_src_number * spin_src_number * time_src_number );
  
  count = nrow * ncol * 2;  /* number of real entries in perambulator matrix */


  /* allocate buffer */
  buffer = (double*)malloc(count*sizeof(double));
  if(buffer == NULL) {
    fprintf(stderr, "[read_perambulator] Error from malloc, could not allocate buffer\n");
    return(5);
  }

  sprintf(tag, "TsoB%.4d.VsoI%.4d.DsoF%d.TsiF%.4d.SsiF%u.DsiF%d.CsiF%d.%s" , peram->nt_src, peram->nv_src, peram->ns_src, peram->nt_snk, VOL3, peram->ns_snk, peram->nc_snk, peram->snk_type);
  fprintf(stdout, "# [read_perambulator] using tag = %s\n", tag);

  sprintf(filename, "%s/%s_quark/cnfg%d/rnd_vec_%d/%s.rndvecnb%.2d.%s.%s.%.5d", \
      laphs_perambulator_path_prefix, peram->quark_type, Nconf, peram->irnd, laphs_perambulator_file_prefix, peram->irnd, peram->quark_type, tag, Nconf);

  fprintf(stdout, "# [read_perambulator] trying to read %lu bytes from file %s\n", count*8, filename);

  ifs = fopen(filename, "rb");
  if (ifs == NULL) {
    fprintf(stderr, "[read_perambulator] Error, could not open file %s\n", filename);
    return(2);
  }

  /* read into buffer */
  items = fread( (void*)buffer, sizeof(double), count, ifs);
  if(items != count) {
    fprintf(stderr, "[read_perambulator] Error, read %lu items, expected %lu\n", items, count);
    return(3);
  }
  fclose(ifs);

  /****************************************************************
   * order the perambulator data in an advantageous way
   ****************************************************************/
  for(t1     = 0; t1     < time_snk_number; ++t1) {
  for(ev1    = 0; ev1    < evec_snk_number; ++ev1) {
  for(dirac1 = 0; dirac1 < spin_snk_number; ++dirac1) {

    for(t2     = 0; t2     < time_src_number; ++t2) {
    for(ev2    = 0; ev2    < evec_src_number; ++ev2) {
    for(dirac2 = 0; dirac2 < spin_src_number; ++dirac2){

      i1 = ( t1 * evec_snk_number + ev1 ) * spin_snk_number + dirac1;
      i2 = ( t2 * evec_src_number + ev2 ) * spin_src_number + dirac2;

      k1 = ( t1 * spin_snk_number + dirac1 ) * evec_snk_number + ev1;
      k2 = ( t2 * spin_src_number + dirac2 ) * evec_src_number + ev2;
      peram->p[2*(k2 * nrow + k1)  ] = buffer[2*(i1 * ncol + i2)  ];
      peram->p[2*(k2 * nrow + k1)+1] = buffer[2*(i1 * ncol + i2)+1];
    }  /* of loop on dirac2 */
    }  /* of loop on ev2 */
    }  /* of loop on t2 */
  }    /* of loop on dirac1 */
  }    /* of loop on ev1 */
  }    /* of loop on t1 */

  free(buffer);
  return(0);
}  /* end of read_perambulator */

/************************************************************************************
 * read eigenvectors and eigenvalues
 ************************************************************************************/

int read_eigensystem(eigensystem_type *es ) {

  unsigned int it;
  int status;

  if(es == NULL) {
    fprintf(stderr, "# [read_eigensystem] Error, eigensystem is NULL vector\n");
    return(1);
  }

  for(it=0; it<es->nt; it++) {
    status = read_eigensystem_timeslice(es, it);
    if(status != 0 ) {
      fprintf(stderr, "# [read_eigensystem] Error from read_eigensystem_timeslice for t = %u\n", it);
      return(2);
    }

    status = read_eigenvalue_timeslice(es, it);
    if(status != 0 ) {
      fprintf(stderr, "# [read_eigensystem] Error from read_eigenvalue_timeslice for t = %u\n", it);
      return(3);
    }

    status = read_phase_timeslice(es, it);
    if(status != 0 ) {
      fprintf(stderr, "# [read_eigensystem] Error from read_phase_timeslice for t = %u\n", it);
      return(4);
    }


  }

  return(0);
} /* end of read_eigensystem */


int read_eigensystem_timeslice(eigensystem_type *es, unsigned int it) {

  char filename[400];
  FILE *ifs=NULL;
  size_t items, count;
  unsigned int nv = es->nv;

  if(it >= es->nt) {
    fprintf(stderr, "# [read_eigensystem_timeslice] Error, t-value too large; %d >= %d\n", it, es->nt);
    return(1);
  }

  if(laphs_eigenvector_number != nv) {
    fprintf(stderr, "# [read_eigensystem_timeslice] Warning local eignevector number differs from global eigenvector number: %d != %d\n", es->nv, laphs_eigenvector_number);
    return(2);
  }

  count = (size_t)nv * (size_t)(LX*LY*LZ) * 6;

  sprintf(filename, "%s/hyp_%.3d_%.3d_%d/nev_%d/%s.%.4d.%.3d", \
      laphs_eigenvector_path_prefix, (int)(alpha_hyp[0]*100), (int)(alpha_hyp[1]*100), N_hyp, nv, laphs_eigenvector_file_prefix,  Nconf, it);
  fprintf(stdout, "# [read_eigensystem_timeslice] reading eigenvectors from file %s\n", filename);

  ifs = fopen(filename, "rb");
  if(ifs == NULL) {
    fprintf(stderr, "[read_eigensystem_timeslice] Error, could not open file %s for reading\n", filename);
    return(1);
  }

  items = fread(es->v[it][0], sizeof(double), count, ifs );
  if(items != count) {
    fprintf(stderr, "[read_eigensystem_timeslice] Error, read %lu items, expected %lu\n", items, count);
    return(3);
  }

  fclose(ifs); ifs = NULL;

  return(0);
}  /* end of read_eigensystem_timeslice */

/*************************************************************************************************************/
int read_eigenvalue_timeslice(eigensystem_type *es, unsigned int it) {

  char filename[400];
  FILE *ifs=NULL;
  size_t items, count;
  unsigned int nv = es->nv;

  if(it >= es->nt) {
    fprintf(stderr, "# [read_eigenvalue_timeslice] Error, t-value too large; %d >= %d\n", it, es->nt);
    return(1);
  }

  if(laphs_eigenvector_number != nv) {
    fprintf(stderr, "# [read_eigenvalue_timeslice] Warning local eignevector number differs from global eigenvector number: %d != %d\n", es->nv, laphs_eigenvector_number);
    return(2);
  }


  count = (size_t)nv;

  sprintf(filename, "%s/hyp_%.3d_%.3d_%d/nev_%d/%s.%.4d.%.3d", \
      laphs_eigenvector_path_prefix, (int)(alpha_hyp[0]*100), (int)(alpha_hyp[1]*100), N_hyp, nv, laphs_eigenvalue_file_prefix,  Nconf, it);
  fprintf(stdout, "# [read_eigenvalue_timeslice] reading eigenvectors from file %s\n", filename);

  ifs = fopen(filename, "rb");
  if(ifs == NULL) {
    fprintf(stderr, "[read_eigenvalue_timeslice] Error, could not open file %s for reading\n", filename);
    return(1);
  }

  items = fread(es->eval+it*nv, sizeof(double), count, ifs );
  if(items != count) {
    fprintf(stderr, "[read_eigenvalue_timeslice] Error, read %lu items, expected %lu\n", items, count);
    return(3);
  }

  fclose(ifs); ifs = NULL;

  return(0);
}  /* end of read_eigenvalue_timeslice */

int read_phase_timeslice(eigensystem_type *es, unsigned int it) {

  char filename[400];
  FILE *ifs=NULL;
  size_t items, count;
  unsigned int nv = es->nv;

  if(it >= es->nt) {
    fprintf(stderr, "# [read_phase_timeslice] Error, t-value too large; %d >= %d\n", it, es->nt);
    return(1);
  }

  if(laphs_eigenvector_number != nv) {
    fprintf(stderr, "# [read_phase_timeslice] Warning local eignevector number differs from global eigenvector number: %d != %d\n", es->nv, laphs_eigenvector_number);
    return(2);
  }

                
  count = (size_t)nv;

  sprintf(filename, "%s/hyp_%.3d_%.3d_%d/nev_%d/%s.%.4d.%.3d", \
    laphs_eigenvector_path_prefix, (int)(alpha_hyp[0]*100), (int)(alpha_hyp[1]*100), N_hyp, nv, laphs_phase_file_prefix,  Nconf, it);
  fprintf(stdout, "# [read_phase_timeslice] reading eigenvectors from file %s\n", filename);

  ifs = fopen(filename, "rb");
  if(ifs == NULL) {
    fprintf(stderr, "[read_phase_timeslice] Error, could not open file %s for reading\n", filename);
    return(1);
  }

  items = fread(es->phase+it*nv, sizeof(double), count, ifs );
  if(items != count) {
    fprintf(stderr, "[read_phase_timeslice] Error, read %lu items, expected %lu\n", items, count);
    return(3);
  }

  fclose(ifs); ifs = NULL;
  return(0);
}  /* end of read_phase_timeslice */




/************************************************************************************
 * read random vector
 ************************************************************************************/
int read_randomvector(randomvector_type *rv, char*quark_type, int irnd) {

  char filename[400];
  FILE *ifs=NULL;
  size_t items, count;

  count = (size_t)laphs_eigenvector_number * (size_t)T * 4 * 2;

  if( rv->rvec == NULL) {
    fprintf(stderr, "[read_randomvector] Error, randomvector field is NULL\n");
    return(1);
  }

  sprintf(filename, "%s/%s_quark/cnfg%d/rnd_vec_%d/%s.rndvecnb%.2d.%s.nbev%.4d.%.4d", \
      laphs_randomvector_path_prefix, quark_type, Nconf, irnd, laphs_randomvector_file_prefix, irnd, quark_type, laphs_eigenvector_number, Nconf);
  fprintf(stdout, "# [read_randomvector] reading eigenvectors from file %s\n", filename);

  ifs = fopen(filename, "rb");
  if(ifs == NULL) {
    fprintf(stderr, "[read_randomvector] Error, could not open file %s for reading\n", filename);
    return(1);
  }

  items = fread(rv->rvec, sizeof(double), count, ifs );
  if(items != count) {
    fprintf(stderr, "[read_randomvector] Error, read %lu items, expected %lu\n", items, count);
    return(2);
  }

  fclose(ifs); ifs = NULL;

  return(0);
}  /* end of read_randomvector */

}
