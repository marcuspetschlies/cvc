#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "global.h"
#include "global.h"
#include "uwerr.h"

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
#if 0
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

  double const z1 ((double*)param)[ iz1 ];
  double const z2 ((double*)param)[ iz2 ];
  double const v  ((double*)param)[ iv  ];
  double const y1 ((double*)param)[ iy1 ];
  double const y2 ((double*)param)[ iy2 ];
  double const w  ((double*)param)[ iw  ];

  double const x       = ( y1 + y2 ) / w * 0.5;
  double const dacosh  = 1 / sqrt ( x*x - 1. );

  /* derivative w.r.t. zaehler1 */
  v_out[zaehler1] =  0.5 / ((double*)param)[nenner] * dacoshx;

  /* derivative w.r.t. zaehler2 */
  v_out[zaehler2] =  0.5 / ((double*)param)[nenner] * dacoshx;

  /* derivative w.r.t. nenner */
  v_out[nenner]  = -(  ((double*)param)[zaehler1] + ((double*)param)[zaehler2] ) / _SQR( ((double*)param)[nenner] ) * dacoshx * 0.5;

  return(0);
}  /* end of dacosh_ratio_deriv */
#endif  /* of if 0 */

/********************************************************/
/********************************************************/

}  /* end of namespace cvc */
