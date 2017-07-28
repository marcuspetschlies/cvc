/* cvc_complex.h */
#ifndef _CVC_COMPLEX_H
#define _CVC_COMPLEX_H

#include <math.h>

/* A complex number. */
/* if complex is define as a macro in complex.h */
#ifdef complex
#undef complex
#endif

namespace cvc 
{

typedef struct {
  double re, im;
} complex;

/* c1 = 0*/
#define _co_eq_zero(c1) {\
  (c1)->re = 0.;\
  (c1)->im = 0.;}

/* c1 = 1*/
#define _co_eq_one(c1) {\
  (c1)->re = 1.;\
  (c1)->im = 0.;}

/* c1 = i*/
#define _co_eq_i(c1) {\
  (c1)->re = 0.;\
  (c1)->im = 1.;}

/* c1 = c2 * c3 */
#define _co_eq_co_ti_co(c1,c2,c3) {\
  (c1)->re = (c2)->re * (c3)->re - (c2)->im * (c3)->im; \
  (c1)->im = (c2)->re * (c3)->im + (c2)->im * (c3)->re;}

/* c1 = c2 / c3 */
#define _co_eq_co_ti_co_inv(c1,c2,c3) {\
  (c1)->re = ( (c2)->re * (c3)->re + (c2)->im * (c3)->im ) / ( (c3)->re*(c3)->re + (c3)->im*(c3)->im); \
  (c1)->im = ( (c2)->im * (c3)->re - (c2)->re * (c3)->im ) / ( (c3)->re*(c3)->re + (c3)->im*(c3)->im);}

/* c1 = c2 * c3^* */
#define _co_eq_co_ti_co_conj(c1,c2,c3) {\
  (c1)->re =  (c2)->re * (c3)->re + (c2)->im * (c3)->im; \
  (c1)->im = -(c2)->re * (c3)->im + (c2)->im * (c3)->re;}

/* c1 = c2 + c3 */
#define _co_eq_co_pl_co(c1,c2,c3) {\
  (c1)->re = (c2)->re + (c3)->re; \
  (c1)->im = (c2)->im + (c3)->im;}

/* c1 = c2 - c3 */
#define _co_eq_co_mi_co(c1,c2,c3) {\
  (c1)->re = (c2)->re - (c3)->re; \
  (c1)->im = (c2)->im - (c3)->im;}

/* c1 += c2 */
#define _co_pl_eq_co(c1,c2) {\
  (c1)->re += (c2)->re; \
  (c1)->im += (c2)->im;}

/* c1 += c2 *r */
#define _co_pl_eq_co_ti_re(c1,c2,r) {\
  (c1)->re += (c2)->re * (r); \
  (c1)->im += (c2)->im * (r);}

/* c1 += c2^* * r */
#define _co_pl_eq_co_conj_ti_re(c1,c2,r) {\
  (c1)->re += (c2)->re * (r); \
  (c1)->im -= (c2)->im * (r);}

/* c1 -= c2 */
#define _co_mi_eq_co(c1,c2) {\
  (c1)->re -= (c2)->re; \
  (c1)->im -= (c2)->im;}

/* c1 += c2 * c3 */
#define _co_pl_eq_co_ti_co(c1,c2,c3) {\
  (c1)->re += (c2)->re * (c3)->re - (c2)->im * (c3)->im; \
  (c1)->im += (c2)->re * (c3)->im + (c2)->im * (c3)->re;}

/* c1 += c2 * Conj(c3) */
#define _co_pl_eq_co_ti_co_conj(c1,c2,c3) {\
  (c1)->re +=  (c2)->re * (c3)->re + (c2)->im * (c3)->im; \
  (c1)->im += -(c2)->re * (c3)->im + (c2)->im * (c3)->re;}

/* c1 -= c2 * c3 */
#define _co_mi_eq_co_ti_co(c1,c2,c3) {\
  (c1)->re -= (c2)->re * (c3)->re - (c2)->im * (c3)->im; \
  (c1)->im -= (c2)->re * (c3)->im + (c2)->im * (c3)->re;}

/* c1 = c2 */
#define _co_eq_co(c1,c2) {\
  (c1)->re = (c2)->re; \
  (c1)->im = (c2)->im;}

/* c1 = c2^* */
#define _co_eq_co_conj(c1,c2) {\
  (c1)->re =  (c2)->re; \
  (c1)->im = -(c2)->im;}

/* c = r+0i */
#define _co_eq_re(c,r) {\
  (c)->re = r; \
  (c)->im = 0.;}

/* c1 *= r */
#define _co_ti_eq_re(c1,r) {\
  (c1)->re *= r; \
  (c1)->im *= r;}

/* c1 = c2*r */
#define _co_eq_co_ti_re(c1,c2,r) {\
  (c1)->re = (c2)->re*r; \
  (c1)->im = (c2)->im*r;}

}
#endif

