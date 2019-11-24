#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "global.h"
#include "global.h"
#include "uwerr.h"
#include "derived_quantities.h"

namespace cvc {

/********************************************************
 *
 ********************************************************/
int ratio_1_1 ( void * param , void * v_in, double * v_out) {
  int zaehler, nenner;
  int * trange = (int*)v_in;
  zaehler = trange[0];
  nenner  = trange[1];

/*  fprintf(stdout, "# [ratio_1_1] zaehler = %d; nenner = %d\n", zaehler, nenner); */
  *v_out = ((double*)param)[zaehler] / ((double*)param)[nenner];
  return(0);
}  /* end of ratio_1_1 */

/********************************************************
 *
 ********************************************************/
int dratio_1_1(void *param , void *v_in, double *v_out) {
  int zaehler, nenner;
  int *trange = (int*)v_in;
  zaehler = trange[0];
  nenner  = trange[1];

  /* derivative w.r.t. zaehler */
  v_out[zaehler] = 1. / ((double*)param)[nenner];
  /* derivative w.r.t. nenner */
  v_out[nenner] =
    -((double*)param)[zaehler] / ( _SQR( ((double*)param)[nenner] ) );

  return(0);
}  /* end of dratio_1_1 */

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int ratio_1_1_sub ( void * param , void * v_in, double * v_out) {
  int * trange = (int*)v_in;
  int zaehler1 = trange[0];
  int nenner1  = trange[1];
  int zaehler2 = trange[2];
  int nenner2  = trange[3];

  *v_out = ((double*)param)[zaehler1] / ((double*)param)[nenner1] - ((double*)param)[zaehler2] / ((double*)param)[nenner2];
  return(0);
}  /* end of ratio_1_1_sub */

/********************************************************
 *
 ********************************************************/
int dratio_1_1_sub ( void *param , void *v_in, double *v_out) {
  int *trange = (int*)v_in;
  int zaehler1 = trange[0];
  int nenner1  = trange[1];
  int zaehler2 = trange[2];
  int nenner2  = trange[3];

  /* derivative w.r.t. zaehler1 */
  v_out[zaehler1] = 1. / ((double*)param)[nenner1];

  /* derivative w.r.t. nenner1 */
  v_out[nenner1 ] = -((double*)param)[zaehler1] / ( _SQR( ((double*)param)[nenner1] ) );

  /* derivative w.r.t. zaehler2 */
  v_out[zaehler2] = -1. / ((double*)param)[nenner2];

  /* derivative w.r.t. nenner2 */
  v_out[nenner2 ] =  ((double*)param)[zaehler2] / ( _SQR( ((double*)param)[nenner2] ) );

  return(0);
}  /* end of dratio_1_1_sub */

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int log_ratio_1_1 ( void * param , void * v_in, double * v_out) {

  int * trange = (int*)v_in;
  int zaehler = trange[0];
  int nenner  = trange[1];

  *v_out = log ( ((double*)param)[zaehler] / ((double*)param)[nenner] );
  return(0);
}  /* end of log_ratio_1_1 */

/********************************************************
 *
 ********************************************************/
int dlog_ratio_1_1(void *param , void *v_in, double *v_out) {

  int *trange = (int*)v_in;
  int zaehler = trange[0];
  int nenner  = trange[1];

  /* derivative w.r.t. zaehler */
  v_out[zaehler] =  1. / ((double*)param)[zaehler];
  /* derivative w.r.t. nenner */
  v_out[nenner]  = -1. / ((double*)param)[nenner];

  return(0);
}  /* end of dlog_ratio_1_1 */

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int acosh_ratio ( void * param , void * v_in, double * v_out) {
 
  int * trange = (int*)v_in;
  int zaehler1 = trange[0];
  int zaehler2 = trange[1];
  int nenner   = trange[2];

  *v_out = acosh( ( ((double*)param)[zaehler1] + ((double*)param)[zaehler2] ) / ((double*)param)[nenner] * 0.5 );
  return(0);
}  /* end of acosh_ratio */

/********************************************************
 *
 ********************************************************/
int dacosh_ratio ( void *param , void *v_in, double *v_out) {

  int *trange = (int*)v_in;
  int zaehler1 = trange[0];
  int zaehler2 = trange[1];
  int nenner   = trange[2];

  double const x =  ( ((double*)param)[zaehler1] + ((double*)param)[zaehler2] ) / ((double*)param)[nenner] * 0.5;
  double const dacoshx = 1. / sqrt( x*x - 1. );

  /* derivative w.r.t. zaehler1 */
  v_out[zaehler1] =  0.5 / ((double*)param)[nenner] * dacoshx;

  /* derivative w.r.t. zaehler2 */
  v_out[zaehler2] =  0.5 / ((double*)param)[nenner] * dacoshx;

  /* derivative w.r.t. nenner */
  v_out[nenner]  = -(  ((double*)param)[zaehler1] + ((double*)param)[zaehler2] ) / _SQR( ((double*)param)[nenner] ) * dacoshx * 0.5;

  return(0);
}  /* end of dacosh_ratio */

/********************************************************/
/********************************************************/

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int acosh_ratio_deriv ( void * param , void * v_in, double * v_out) {
 
  int * const trange = (int*)v_in;
  int const iz1 = trange[0];
  int const iz2 = trange[1];
  int const iv  = trange[2];

  int const iy1 = trange[3];
  int const iy2 = trange[4];
  int const iw  = trange[5];

  double const z1 = ((double*)param)[ iz1 ];
  double const z2 = ((double*)param)[ iz2 ];
  double const v  = ((double*)param)[ iv  ];
  double const y1 = ((double*)param)[ iy1 ];
  double const y2 = ((double*)param)[ iy2 ];
  double const w  = ((double*)param)[ iw  ];

  double const x       = ( y1 + y2 ) / w * 0.5;
  double const dacoshx = 1 / sqrt ( x*x - 1. );

  *v_out =  ( z1 + z2  - ( y1 + y2 ) * v / w ) / w * 0.5 * dacoshx;
  return(0);
}  /* end of acosh_ratio_deriv */

/********************************************************
 *
 ********************************************************/
int dacosh_ratio_deriv ( void *param , void *v_in, double *v_out) {

  int * const trange = (int*)v_in;
  int const iz1 = trange[0];
  int const iz2 = trange[1];
  int const iv  = trange[2];

  int const iy1 = trange[3];
  int const iy2 = trange[4];
  int const iw  = trange[5];

  double const z1 = ((double*)param)[ iz1 ];
  double const z2 = ((double*)param)[ iz2 ];
  double const v  = ((double*)param)[ iv  ];
  double const y1 = ((double*)param)[ iy1 ];
  double const y2 = ((double*)param)[ iy2 ];
  double const w  = ((double*)param)[ iw  ];

  double const x       = ( y1 + y2 ) / w * 0.5;
  double const dacosh  = 1 / sqrt ( x*x - 1. );

  /* derivative w.r.t. z1 */
  v_out[iz1] =  dacosh / w * 0.5;

  /* derivative w.r.t. z2 */
  v_out[iz2] =  dacosh / w * 0.5;

  /* derivative w.r.t. v */
  v_out[iv]  = -dacosh * ( y1 + y2 )  / ( 2. * w * w );

  /* derivative w.r.t. y1 */
  v_out[iy1]  = (-4 * v * w + (y1 + y2) * (z1 + z2) ) / ( w * (4 * _POW2(w) - _POW2(y1 + y2) )  * sqrt(-4 + _POW2( ( y1 + y2 ) / w) ) );

  /* derivative w.r.t. y2 */
  v_out[iy2]  = (-4 * v * w + ( y1 + y2 ) * ( z1 + z2 ) ) / ( w * ( 4 * _POW2(w) - _POW2(y1 + y2) ) * sqrt(-4 + _POW2( ( y1 + y2 ) / w ) ) );

  /* derivative w.r.t. y2 */
  v_out[iw]  = ( -v * ( y1 + y2 ) * ( -8 * _POW2(w) + _POW2( y1 + y2 ) ) - 4 * _POW3(w) * ( z1 + z2 ) ) \
        / ( _POW3(w) * ( 4 * _POW2(w) - _POW2( y1 + y2 ) ) * sqrt( -4 + _POW2( (y1 + y2) / w ) ) );

  return(0);
}  /* end of dacosh_ratio_deriv */
#if 0
#endif  /* of if 0 */

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int a_mi_b_ti_c ( void * param , void * v_in, double * v_out) {
  int * const trange = (int*)v_in;
  int const idx_a = trange[0];
  int const idx_b = trange[1];
  int const idx_c = trange[2];

  *v_out = ((double*)param)[idx_a] - ((double*)param)[idx_b] * ((double*)param)[idx_c];
  return(0);
}  /* end of a_mi_b_ti_c */

/********************************************************
 *
 ********************************************************/
int da_mi_b_ti_c ( void *param , void *v_in, double *v_out) {
  int * const trange = (int*)v_in;
  int const idx_a = trange[0];
  int const idx_b = trange[1];
  int const idx_c = trange[2];

  /* derivative w.r.t. a */
  v_out[idx_a] = 1.;

  /* derivative w.r.t. b */
  v_out[idx_b] = -((double*)param)[idx_c];

  /* derivative w.r.t. c */
  v_out[idx_c] = -((double*)param)[idx_b];

  return(0);
}  /* end of da_mi_b_ti_c */

/********************************************************
 * 1st cumulant
 * < x >
 ********************************************************/
int cumulant_1  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[0];

  *v_out = ((double*)param)[idx_a];
  return(0);
}  /* end of cumulant_1 */

int dcumulant_1  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[0];

  v_out[idx_a] = 1.;

  return(0);
}  /* end of cumulant_1 */

/********************************************************
 * 2nd cumulant
 * < x^2 > - < x >^2
 ********************************************************/
int cumulant_2  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[1];
  int const idx_b = idx_range[0];

  *v_out = ((double*)param)[idx_a] - _POW2( ((double*)param)[idx_b] );
  return(0);
}  /* end of cumulant_2 */

int dcumulant_2  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[1];
  int const idx_b = idx_range[0];

  v_out[idx_a] = 1.;

  v_out[idx_b] = -2. * ((double*)param)[idx_b];
  return(0);
}  /* end of cumulant_2 */

/********************************************************
 * 3rd cumulant
 * < x^3 > - 3 < x^2 > * < x > + 2 < x >^3 
 ********************************************************/
int cumulant_3  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[2];
  int const idx_b = idx_range[1];
  int const idx_c = idx_range[0];

  *v_out = ((double*)param)[idx_a] - 3 * ((double*)param)[idx_b] *  ((double*)param)[idx_c] + 2 *  _POW3( ((double*)param)[idx_c] );
  return(0);
}  /* end of cumulant_3 */

int dcumulant_3  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[2];
  int const idx_b = idx_range[1];
  int const idx_c = idx_range[0];

  v_out[idx_a] =  1;

  v_out[idx_b] = -3 * ((double*)param)[idx_c];

  v_out[idx_c] = -3 * ((double*)param)[idx_b] + 6. * _POW2( ((double*)param)[idx_c] );
  return(0);
}  /* end of cumulant_3 */

/********************************************************
 * 4th cumulant
 * < x^4 > - 4 < x^3 > * < x >  -3 < x^2 >^2 + 12 < x^2 > * < x >^2 - 6 < x >^4
 ********************************************************/
int cumulant_4  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[3];
  int const idx_b = idx_range[2];
  int const idx_c = idx_range[1];
  int const idx_d = idx_range[0];

  *v_out = 
    +      ((double*)param)[idx_a] 
    -  4 * ((double*)param)[idx_b] *  ((double*)param)[idx_d] + 
    -  3 * _POW2( ((double*)param)[idx_c] ) 
    + 12 * ((double*)param)[idx_c] * _POW2( ((double*)param)[idx_d] )
    -  6 *  _POW4( ((double*)param)[idx_d] );
  return(0);
}  /* end of cumulant_4 */

int dcumulant_4  ( void * param , void * v_in, double * v_out) {
  
  int * const idx_range = (int*)v_in;
  int const idx_a = idx_range[3];
  int const idx_b = idx_range[2];
  int const idx_c = idx_range[1];
  int const idx_d = idx_range[0];

  v_out[idx_a] =  1;

  v_out[idx_b] = -4 * ((double*)param)[idx_d]; 

  v_out[idx_c] = -6. * ((double*)param)[idx_c] + 12 * _POW2( ((double*)param)[idx_d] );

  v_out[idx_d] = -4 * ((double*)param)[idx_b] + 24 * ((double*)param)[idx_c] * ((double*)param)[idx_d] - 24 * _POW3( ((double*)param)[idx_d] );
  return(0);
}  /* end of cumulant_4 */

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
int ratio_1_2_mi_3 ( void * param , void * v_in, double * v_out) {
  int * const trange = (int*)v_in;
  int const i1  = trange[0];
  int const i2  = trange[1];
  int const i3  = trange[1];

  *v_out = ((double*)param)[i1] / ((double*)param)[i2] - ((double*)param)[i3];
  return(0);
}  /* end of ratio_1_2_mi_3 */

/********************************************************
 *
 ********************************************************/
int dratio_1_2_mi_3(void *param , void *v_in, double *v_out) {
  int * const trange = (int*)v_in;
  int const i1 = trange[0];
  int const i2 = trange[1];
  int const i3 = trange[2];

  /* derivative w.r.t. zaehler */
  v_out[i1] = 1. / ((double*)param)[i2];
  /* derivative w.r.t. nenner */
  v_out[i2] = -((double*)param)[i1] / ( _SQR( ((double*)param)[i2] ) );
  /* derivative w.r.t. subtraction */
  v_out[i3] = -1.;

  return(0);
}  /* end of dratio_1_2_mi_3 */

/********************************************************/
/********************************************************/

}  /* end of namespace cvc */
