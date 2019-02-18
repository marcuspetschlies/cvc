#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "incomp_gamma.h"

double incomp_gamma(
		 double const U     /* upper limit of integration, in */,
		 double const Nexp  /* exponent in the integrand */) {
  if(Nexp==1.) {
    return(1-exp(-U));
  } else if(Nexp==0.5) {
    return(erf(sqrt(U)));
  } else {
    return( -pow(U,Nexp-1.) * exp(-U) / tgamma(Nexp) + incomp_gamma(U, Nexp-1.) );
  }
}

