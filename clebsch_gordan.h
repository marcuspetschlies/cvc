#ifndef _CLEBSCH_GORDAN_H
#define _CLEBSCH_GORDAN_H

namespace cvc {

static inline double delta_symbol ( int a2, int b2, int c2 ) {

  int pl_a_pl_b_mi_c = ( a2 + b2 - c2 ) / 2;
  int pl_a_mi_b_pl_c = ( a2 - b2 + c2 ) / 2;
  int mi_a_pl_b_pl_c = (-a2 + b2 + c2 ) / 2;

  int a_pl_b_pl_c_pl_one = ( a2 + b2 + c2 ) / 2 + 1;

  if ( ( pl_a_pl_b_mi_c < 0 ) || ( pl_a_mi_b_pl_c < 0 ) || ( mi_a_pl_b_pl_c < 0 ) || ( a_pl_b_pl_c_pl_one < 0 )) {
    fprintf ( stderr, "[delta_symbol] Error, negative argument to factorial %3d %3d %3d\n", a2, b2, c2 );
    return(-1.);
  }
  
  return ( sqrt( factorial (pl_a_pl_b_mi_c) * factorial(pl_a_mi_b_pl_c) * factorial(mi_a_pl_b_pl_c) / (double)factorial(a_pl_b_pl_c_pl_one) ) );
}  /* end of delta_symbol */

/******************************************************************************************/
/******************************************************************************************/

/******************************************************************************************
 *
 ******************************************************************************************/
static inline double clebsch_gordan_coeff ( int c2, int gamma2, int a2, int alpha2, int b2, int beta2 ) {

  if ( gamma2 != alpha2 + beta2 ) return(0);
  double p = 0.;
  double s = 0.;


    int c_pl_gamma = ( c2 + gamma2 ) / 2;
    int c_mi_gamma = ( c2 - gamma2 ) / 2;
    int c2_pl_one  = c2 + 1;

    int a_pl_alpha = ( a2 + alpha2 ) / 2;
    int a_mi_alpha = ( a2 - alpha2 ) / 2;

    int b_pl_beta  = ( b2 + beta2  ) / 2;
    int b_mi_beta  = ( b2 - beta2  ) / 2;

    p = sqrt( (double)factorial( c_pl_gamma ) * factorial( c_mi_gamma ) * (double)c2_pl_one / ( factorial( a_pl_alpha ) * factorial( a_mi_alpha ) * factorial( b_pl_beta ) * factorial( b_mi_beta ) ) );

    // fprintf ( stdout, "# [] a2 %d alpha2 %d b2 %d beta2  %d c2 %d gamma2 %d c + gamma %d c - gamma %d c2 + 1 %d a + alpha %d a - alpha %d b + beta %d b - beta %d p %16.9f\n",
    //   a2, alpha2, b2, beta2, c2, gamma2, c_pl_gamma, c_mi_gamma, c2_pl_one, a_pl_alpha, a_mi_alpha, b_pl_beta, b_mi_beta, p );



    int c_pl_b_pl_alpha = ( c2 + b2 + alpha2 ) / 2;
    // int a_mi_alpha      = ( a2 - alpha2 ) / 2;

    int c_mi_a_pl_b = ( c2 - a2 + b2 ) / 2;
    // int c_pl_gamma  = ( c2 + gamma2 ) / 2;
    int a_mi_b_mi_gamma = ( a2 - b2 - gamma2 ) / 2;

    // z <= c_pl_b_pl_alpha
    // z <= c_mi_a_pl_b
    // z <= c_pl_gamma
    //
    // z >= 0
    // z >= - ( a_mi_alpha )
    // z >= -( a_mi_b_mi_gamma )

    // fprintf ( stdout, "# [] c_pl_b_pl_alpha %d a_mi_alpha %d c_mi_a_pl_b %d c_pl_gamma %d a_mi_b_mi_gamma %d\n", c_pl_b_pl_alpha, a_mi_alpha, c_mi_a_pl_b, c_pl_gamma, a_mi_b_mi_gamma );

    int zmin = _MAX( _MAX( 0, -a_mi_alpha), -a_mi_b_mi_gamma);
    
    int zmax = _MIN( _MIN( c_pl_b_pl_alpha, c_mi_a_pl_b), c_pl_gamma );

    // fprintf ( stdout, "# [] zmin %d zmax %d\n", zmin, zmax );

    for ( int z = zmin; z <= zmax; z++ ) {
      int zsign = ( b_pl_beta + z)%2==0 ? 1 : -1;
      double numerator   = factorial( c_pl_b_pl_alpha - z ) * factorial( a_mi_alpha + z );
      double denominator = factorial( z ) * factorial ( c_mi_a_pl_b - z) * factorial ( c_pl_gamma - z ) * factorial ( a_mi_b_mi_gamma + z );

      s += zsign * numerator / denominator;
      // fprintf ( stdout, "# [] z %d zsign %d num %16.9f den %16.9f s = %16.9f\n", z, zsign, numerator, denominator, s );
    }


    double ds = delta_symbol (a2,b2,c2);

    // fprintf ( stdout, "# [] d = %16.9f p = %16.9f s = %16.9f\n", ds, p, s );


  return ( ds * p * s );
}  /* end of clebsch_gordan_coeff */


}
#endif
