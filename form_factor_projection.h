/************************************************
 * form_factor_projection.h
 *
 * Thu Jun  1 09:20:59 CEST 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/
#ifndef _FORM_FACTOR_PROJECTION_H
#define _FORM_FACTOR_PROJECTION_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>


namespace cvc {

static inline void _project_pseudotensor_to_scalar (double _Complex **_h, double _Complex *_s, double _p[4], double _q[4]) {
  double _plat[4] = { 2.*sin( 0.5 * (_p)[0] ), 2.*sin( 0.5 * (_p)[1] ), 2.*sin( 0.5 * (_p)[2] ), 2.*sin( 0.5 * (_p)[3] ) };
  double _qlat[4] = { 2.*sin( 0.5 * (_q)[0] ), 2.*sin( 0.5 * (_q)[1] ), 2.*sin( 0.5 * (_q)[2] ), 2.*sin( 0.5 * (_q)[3] ) };
  double _pp = _plat[0] * _plat[0] + _plat[1] * _plat[1] + _plat[2] * _plat[2] + _plat[3] * _plat[3];
  double _qq = _qlat[0] * _qlat[0] + _qlat[1] * _qlat[1] + _qlat[2] * _qlat[2] + _qlat[3] * _qlat[3];
  double _pq = _plat[0] * _qlat[0] + _plat[1] * _qlat[1] + _plat[2] * _qlat[2] + _plat[3] * _qlat[3];
  (*_s) = 0.;
  (*_s) += (_h)[0][1] * _plat[2] * _qlat[3];
  (*_s) -= (_h)[0][1] * _plat[3] * _qlat[2];
  (*_s) -= (_h)[0][2] * _plat[1] * _qlat[3];
  (*_s) += (_h)[0][2] * _plat[3] * _qlat[1];
  (*_s) += (_h)[0][3] * _plat[1] * _qlat[2];
  (*_s) -= (_h)[0][3] * _plat[2] * _qlat[1];
  (*_s) -= (_h)[1][0] * _plat[2] * _qlat[3];
  (*_s) += (_h)[1][0] * _plat[3] * _qlat[2];
  (*_s) += (_h)[1][2] * _plat[0] * _qlat[3];
  (*_s) -= (_h)[1][2] * _plat[3] * _qlat[0];
  (*_s) -= (_h)[1][3] * _plat[0] * _qlat[2];
  (*_s) += (_h)[1][3] * _plat[2] * _qlat[0];
  (*_s) += (_h)[2][0] * _plat[1] * _qlat[3];
  (*_s) -= (_h)[2][0] * _plat[3] * _qlat[1];
  (*_s) -= (_h)[2][1] * _plat[0] * _qlat[3];
  (*_s) += (_h)[2][1] * _plat[3] * _qlat[0];
  (*_s) += (_h)[2][3] * _plat[0] * _qlat[1];
  (*_s) -= (_h)[2][3] * _plat[1] * _qlat[0];
  (*_s) -= (_h)[3][0] * _plat[1] * _qlat[2];
  (*_s) += (_h)[3][0] * _plat[2] * _qlat[1];
  (*_s) += (_h)[3][1] * _plat[0] * _qlat[2];
  (*_s) -= (_h)[3][1] * _plat[2] * _qlat[0];
  (*_s) -= (_h)[3][2] * _plat[0] * _qlat[1];
  (*_s) += (_h)[3][2] * _plat[1] * _qlat[0];
  (*_s) /= _pp * _qq - _pq * _pq;
}  /* end of _project_pseudotensor_to_scalar */

static inline void _project_tensor_to_scalar (double _Complex **h, double _Complex *h_t, double _Complex *h_l, double p[4]) {

  double plat[4] = { 2. * sin(0.5 * p[0]) , 2. * sin(0.5 * p[1]) , 2. * sin(0.5 * p[2]) , 2. * sin(0.5 * p[3]) };

  double plat2inv = plat[0] * plat[0] + plat[1] * plat[1] + plat[2] * plat[2] + plat[3] * plat[3];
  plat2inv = plat2inv == 0. ? 0. : 1./plat2inv;

  double _Complex _h_l = plat[0] * ( plat[0] * h[0][0] + plat[1] * h[1][0] + plat[2] * h[2][0] + plat[3] * h[3][0] )
                       + plat[1] * ( plat[0] * h[0][1] + plat[1] * h[1][1] + plat[2] * h[2][1] + plat[3] * h[3][1] )
                       + plat[2] * ( plat[0] * h[0][2] + plat[1] * h[1][2] + plat[2] * h[2][2] + plat[3] * h[3][2] )
                       + plat[3] * ( plat[0] * h[0][3] + plat[1] * h[1][3] + plat[2] * h[2][3] + plat[3] * h[3][3] );
            
  _h_l *=  plat2inv == 0. ? 1. : plat2inv;

  double _Complex _h_t = h[0][0] + h[1][1] + h[2][2] + h[3][3] - _h_l;

  if ( h_l != NULL ) { h_l[0] = _h_l; }
  if ( h_t != NULL ) { h_t[0] = _h_t; }
}  /* end of project_tensor_to_scalar */


}  /* end of namespace cvc */

#endif
