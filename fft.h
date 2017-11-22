/************************************************
 * fft.h
 ************************************************/
#ifndef _HAVE_FFT_H
#define _HAVE_FFT_H

namespace cvc {

void complex_field_reorder ( double _Complex *r, double _Complex *s, unsigned int *p, unsigned int N);

int ft_4dim ( double *r, double *s, int sign, int init );

int half_link_momentum_phase_4dim ( double *r, double *s, int sign, int mu, int init );

}  /* end of namespace cvc */

#endif
