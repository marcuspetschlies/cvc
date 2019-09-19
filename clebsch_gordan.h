#ifndef _CLEBSCH_GORDAN_H
#define _CLEBSCH_GORDAN_H

namespace cvc {

static inline double delta_symbol ( int const a2, int const b2, int const c2 ) {

  int const pl_a_pl_b_mi_c = ( a2 + b2 - c2 ) / 2;
  int const pl_a_mi_b_pl_c = ( a2 - b2 + c2 ) / 2;
  int const mi_a_pl_b_pl_c = (-a2 + b2 + c2 ) / 2;

  int const a_pl_b_pl_c_pl_one = ( a2 + b2 + c2 ) / 2 + 1;

  if ( ( pl_a_pl_b_mi_c < 0 ) || ( pl_a_mi_b_pl_c < 0 ) || ( mi_a_pl_b_pl_c < 0 ) || ( a_pl_b_pl_c_pl_one < 0 )) {
    fprintf ( stderr, "[delta_symbol] Error, negative argument to factorial %3d %3d %3d\n", a2, b2, c2 );
    return(-1.);
  }
  
  return ( sqrt( factorial (pl_a_pl_b_mi_c) * factorial(pl_a_mi_b_pl_c) * factorial(mi_a_pl_b_pl_c) / (double)factorial(a_pl_b_pl_c_pl_one) ) );
}  /* end of delta_symbol */

/******************************************************************************************/
/******************************************************************************************/

/******************************************************************************************
 * calculate Clebsch-Gordan coefficients
 *
 * IN
 * c2     = 2 J_3
 * gamma2 = 2 M_3
 *
 * a2     = 2 J_1
 * alpha2 = 2 M_1
 *
 * b2     = 2 J_2
 * beta   = 2 M_2
 *
 * OUT
 * < J_3, M_3 | J_1, M_1; J_2, M_2 >
 * 
 ******************************************************************************************/
static inline double clebsch_gordan_coeff ( int const c2, int const gamma2, int const a2, int const alpha2, int const b2, int const beta2 ) {

  if ( gamma2 != alpha2 + beta2 ) return(0);
  double p = 0.;
  double s = 0.;


    int const c_pl_gamma = ( c2 + gamma2 ) / 2;
    int const c_mi_gamma = ( c2 - gamma2 ) / 2;
    int const c2_pl_one  = c2 + 1;

    int const a_pl_alpha = ( a2 + alpha2 ) / 2;
    int const a_mi_alpha = ( a2 - alpha2 ) / 2;

    int const b_pl_beta  = ( b2 + beta2  ) / 2;
    int const b_mi_beta  = ( b2 - beta2  ) / 2;

    p = sqrt( (double)factorial( c_pl_gamma ) * factorial( c_mi_gamma ) * (double)c2_pl_one / ( factorial( a_pl_alpha ) * factorial( a_mi_alpha ) * factorial( b_pl_beta ) * factorial( b_mi_beta ) ) );

    // fprintf ( stdout, "# [] a2 %d alpha2 %d b2 %d beta2  %d c2 %d gamma2 %d c + gamma %d c - gamma %d c2 + 1 %d a + alpha %d a - alpha %d b + beta %d b - beta %d p %16.9f\n",
    //   a2, alpha2, b2, beta2, c2, gamma2, c_pl_gamma, c_mi_gamma, c2_pl_one, a_pl_alpha, a_mi_alpha, b_pl_beta, b_mi_beta, p );



    int const c_pl_b_pl_alpha = ( c2 + b2 + alpha2 ) / 2;
    // int a_mi_alpha      = ( a2 - alpha2 ) / 2;

    int const c_mi_a_pl_b = ( c2 - a2 + b2 ) / 2;
    // int c_pl_gamma  = ( c2 + gamma2 ) / 2;
    int const a_mi_b_mi_gamma = ( a2 - b2 - gamma2 ) / 2;

    // z <= c_pl_b_pl_alpha
    // z <= c_mi_a_pl_b
    // z <= c_pl_gamma
    //
    // z >= 0
    // z >= - ( a_mi_alpha )
    // z >= -( a_mi_b_mi_gamma )

    // fprintf ( stdout, "# [] c_pl_b_pl_alpha %d a_mi_alpha %d c_mi_a_pl_b %d c_pl_gamma %d a_mi_b_mi_gamma %d\n", c_pl_b_pl_alpha, a_mi_alpha, c_mi_a_pl_b, c_pl_gamma, a_mi_b_mi_gamma );

    int const zmin = _MAX( _MAX( 0, -a_mi_alpha), -a_mi_b_mi_gamma);
    
    int const zmax = _MIN( _MIN( c_pl_b_pl_alpha, c_mi_a_pl_b), c_pl_gamma );

    // fprintf ( stdout, "# [] zmin %d zmax %d\n", zmin, zmax );

    for ( int z = zmin; z <= zmax; z++ ) {
      int const zsign = ( b_pl_beta + z)%2==0 ? 1 : -1;
      double const numerator   = factorial( c_pl_b_pl_alpha - z ) * factorial( a_mi_alpha + z );
      double const denominator = factorial( z ) * factorial ( c_mi_a_pl_b - z) * factorial ( c_pl_gamma - z ) * factorial ( a_mi_b_mi_gamma + z );

      s += zsign * numerator / denominator;
      // fprintf ( stdout, "# [] z %d zsign %d num %16.9f den %16.9f s = %16.9f\n", z, zsign, numerator, denominator, s );
    }


    double const ds = delta_symbol (a2,b2,c2);

    // fprintf ( stdout, "# [] d = %16.9f p = %16.9f s = %16.9f\n", ds, p, s );


  return ( ds * p * s );
}  /* end of clebsch_gordan_coeff */

/******************************************************************************************
 * C matrix from Luescher's Quantization condition, numerical values
 *
 * IN
 * l1 = L_1, m1 = M_1
 * l2 = L_2, m2 = M_2
 * l3 = L_3, m3 = M_3
 *
 * OUT
 * cf. Bernard et al. JHEP08(2008)024
 * ( cf. also formula (2.25) in Goeckeler et al. PRD 86 094513 (2012) )
 *
 ******************************************************************************************/
static inline double _Complex luescher_c_matrix ( int const l1, int const m1, int const l2, int const m2, int const l3, int const m3 ) {

  double _Complex const ipow[4] = { 1., I, -1., -I. };
  double const norm = sqrt( ( 2. * l1 + 1. ) * ( 2. * l2 + 1.  ) / ( 2. * l3 + 1. ) );

  double const c1 = clebsch_gordan_coeff ( 2*l3, 0, 2*l1, 0, 2*l2, 0 );

  double const c2 = clebsch_gordan_coeff ( 2*l3, 2*m3, 2*l1, 2*m1, 2*l2, 2*m2 );

  return ( ipow[ ( ( l1 - l2 + l3 ) % 4 +4  ) % 4 ] * norm * c1 * c2 ):

}  /* end of luescher_c_matrix */

/******************************************************************************************
 * C matrix from Luescher's Quantization condition, numerical values
 *
 * IN
 * l1 = L_1, m1 = M_1
 * l2 = L_2, m2 = M_2
 * l3 = L_3, m3 = M_3
 *
 * OUT
 * cf. Bernard et al. JHEP08(2008)024
 * ( cf. also formula (2.25) in Goeckeler et al. PRD 86 094513 (2012) )
 *
 ******************************************************************************************/
static inline double _Complex cg_lm_spin_12 ( int const l2, int const m2, int const sigma2, int const j2, int const s2 ) {
  return ( clebsch_gordan_coeff ( j2, s2, l2, m2, 1, sigma2 ) );
}  /* end of cg_lm_spin_12 */

}
#endif
