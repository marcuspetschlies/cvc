#ifndef _PM_H
#define _PM_H

namespace cvc {

/***************************************************************************
 * x <- gamma * y
 ***************************************************************************/
inline void pm_eq_gamma_ti_pm ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y ) {

  for ( int ll = 0; ll < 12; ll++ ) {
    
    for ( int i = 0; i < 4; i++ ) {
    
      for (int kc = 0; kc < 3; kc++ ) {

        int const ii = 3 * i + kc;
    
        double _Complex z = 0.;

        for ( int k = 0; k < 4; k++ ) {

          int const kk = 3 * k + kc;

          z += g->m[i][k] * y[kk][ll];

        }  /* end of loop on k */

        x[ii][ll] = z;

      }
    }
  }
  return;
}  /* end of pm_eq_gamma_ti_pm */

/***************************************************************************
 * x <- gamma * y^+
 ***************************************************************************/
inline void pm_eq_gamma_ti_pm_dag ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y ) {

  for ( int ll = 0; ll < 12; ll++ ) {
    
    for ( int i = 0; i < 4; i++ ) {
    
      for (int kc = 0; kc < 3; kc++ ) {

        int const ii = 3 * i + kc;
    
        double _Complex z = 0.;

        for ( int k = 0; k < 4; k++ ) {

          int const kk = 3 * k + kc;

          z += g->m[i][k] * conj( y[ll][kk] );

        }  /* end of loop on k */

        x[ii][ll] = z;

      }
    }
  }
  return;
}  /* end of pm_eq_gamma_ti_pm_dag */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += y[ii][kk] * z[kk][ll];
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_ti_pm */

/***************************************************************************
 * x <- a x y + b * z
 ***************************************************************************/
inline void pm_eq_pm_pl_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z, double _Complex const a, double _Complex const b ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    x[ii][ll] = a * y[ii][ll] + b * z[ii][ll];
  }}
  return;
}  /* pm_eq_pm_pl_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline double _Complex co_eq_tr_pm ( double _Complex ** const y ) {

  double _Complex x = 0.;

  for ( int ii = 0; ii < 12; ii++ ) {
    x += y[ii][ii];
  }

  return ( x );
}  /* end of co_eq_tr_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_dag_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += conj ( y[kk][ii] ) * z[kk][ll];
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_dag_ti_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_pm_dag ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += y[ii][kk] * conj ( z[ll][kk] );
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_ti_pm_dag */

/***************************************************************************
 * x += gamma * y * gamma
 ***************************************************************************/
inline void pm_pl_eq_gamma_ti_pm_ti_gamma ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y, double _Complex const a ) {

  for ( int j = 0; j < 4; j++ ) {
    for ( int jc = 0; jc < 3; jc++ ) {

      int const jj = 3 * j + jc;

      for ( int i = 0; i < 4; i++ ) {
    
        for (int ic = 0; ic < 3; ic++ ) {
        
          int const ii = 3 * i + ic;

          double _Complex z = 0.;

          for ( int l = 0; l < 4; l++ ) {

            int const ll = 3 * l + jc;

            for ( int k = 0; k < 4; k++ ) {

              int const kk = 3 * k + ic;

              z += g->m[i][k] * y[kk][ll] * g->m[l][j];

            }  /* end of loop on k */

          }  /* end of loop on l */

          x[ii][jj] = a * x[ii][jj] + z;

        }  /* ic */
      }  /* i */
    }  /* jc */
  }  /* j */

  return;

}  /* end of pm_pl_eq_gamma_ti_pm_ti_gamma */

/***************************************************************************
 * x += gamma * y^dag * gamma
 *
 * NOTE: x and y MUST BE DIFFERENT memory locations
 ***************************************************************************/
inline void pm_pl_eq_gamma_ti_pm_dag_ti_gamma ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y, double _Complex const a ) {

  for ( int j = 0; j < 4; j++ ) {
    for ( int jc = 0; jc < 3; jc++ ) {

      int const jj = 3 * j + jc;

      for ( int i = 0; i < 4; i++ ) {
    
        for (int ic = 0; ic < 3; ic++ ) {
        
          int const ii = 3 * i + ic;

          double _Complex z = 0.;

          for ( int l = 0; l < 4; l++ ) {

            int const ll = 3 * l + jc;

            for ( int k = 0; k < 4; k++ ) {

              int const kk = 3 * k + ic;

              z += g->m[i][k] * conj( y[ll][kk] ) * g->m[l][j];

            }  /* end of loop on k */

          }  /* end of loop on l */

          x[ii][jj] = a * x[ii][jj] + z;

        }  /* ic */
      }  /* i */
    }  /* jc */
  }  /* j */

  return;

}  /* end of pm_pl_eq_gamma_ti_pm_dag_ti_gamma */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_co  ( double _Complex ** const x , double _Complex ** const y, double _Complex const a ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    x[ii][ll] = y[ii][ll] * a;
  }}
  return;
}

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_set_from_sf_point ( double _Complex ** const p, double ** const s, unsigned int ix ) {

  unsigned int const iix = _GSI( ix );

  for ( int k=0; k<12; k++) {
    double * const _prop = s[k] + iix;
    for ( int i=0; i<12;i++) {
      p[i][k] = _prop[2 * i ] + I * _prop[2 * i + 1];
    }
  }

return;
}  /* pm_set_from_sf */


/***************************************************************************
 *
 ***************************************************************************/
inline void pm_print (double _Complex ** const p, char * const name, FILE * ofs ) {
  fprintf ( ofs, "%s <- array( dim=c(12,12)) \n", name );
  for( int i = 0; i < 12; i++ ) {
  for( int k = 0; k < 12; k++ ) {
    fprintf ( ofs, "%s[%d,%d] <- %25.16e + %25.16e*1.i\n", name, i+1, k+1, creal( p[i][k] ), cimag ( p[i][k] ) );
  }}
  return;
}  /* pm_print */


/***************************************************************************
 * x <- a y^+ + b z
 ***************************************************************************/
inline void pm_eq_pm_dag_pl_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z, double _Complex const a, double _Complex const b ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    x[ii][ll] = a * conj ( y[ll][ii] ) + b * z[ii][ll];
  }}
  return;
}  /* pm_eq_pm_dag_pl_pm */


}

#endif
