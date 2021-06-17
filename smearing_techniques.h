/**********************************
 * smearing_techniques.h  
 *
 * originally:
 * Author: Marc Wagner
 * Date: September 2007
 *
 * February 2010
 * taken over to cvc_parallel, parallelized
 *
 * Fri Dec  9 09:24:13 CET 2016
 * now copied from cvc_parallel_5d
 *
 **********************************/
#ifndef _SMEARING_TECHNIQUES_H
#define _SMEARING_TECHNIQUES_H

#include "cvc_linalg.h"

namespace cvc {

int APE_Smearing(double *smeared_gauge_field, double const APE_smearing_alpha, int const APE_smearing_niter);

int Jacobi_Smearing(double *smeared_gauge_field, double *psi, int N, double kappa);

int rms_radius ( double *r_rms, double *s, int source_coords[4] );

int source_profile ( double *s, int source_coords[4], char*prefix );


}
#endif
