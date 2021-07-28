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

int rms_radius ( double ** const r2, double ** const w2, double * const s, int const source_coords[4] );

int source_profile ( double *s, int source_coords[4], char*prefix );

void _ADD_STAPLES_TO_COMPONENT( double * const buff_out, double * const buff_in, unsigned int const x,  int const to, int const via);

void exposu3( double * const vr, double * const p );

void generic_staples ( double * const buff_out, const unsigned int x, const int mu, double * const buff_in );

int stout_smear_inplace ( double * const m_field, const int stout_n, const double stout_rho, double * const buffer );


}
#endif
