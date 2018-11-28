#ifndef _TABLE_INIT_D_H
#define _TABLE_INIT_D_H

/****************************************************
 * table_init_d.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline double * init_1level_dtable ( size_t const N0 ) {
  return( N0 == 0 ? NULL : ( double *) calloc ( N0 , sizeof( double ) ) );
}  // end of init_1level_dtable

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_dtable ( double **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_dtable] active\n");
  *s = NULL;
}  // end of fini_1level_dtable

/************************************************************************************/
/************************************************************************************/

inline double ** init_2level_dtable (size_t const N0, size_t const N1 ) {
  double * s__ = NULL;
  s__ = init_1level_dtable ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  double ** s_ = ( N0 == 0 ) ? NULL : ( double **) malloc( N0 * sizeof( double *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_dtable ( double *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_dtable] active\n");
    fini_1level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_dtable

/************************************************************************************/
/************************************************************************************/


inline double *** init_3level_dtable (size_t const N0, size_t const N1, size_t const N2 ) {
  double ** s__ = NULL;
  s__ = init_2level_dtable ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  double *** s_ = ( N0 == 0 ) ? NULL : ( double ***) malloc( N0 * sizeof( double **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_dtable ( double **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_dtable] active\n");
    fini_2level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_dtable

/************************************************************************************/
/************************************************************************************/


inline double **** init_4level_dtable (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  double *** s__ = NULL;
  s__ = init_3level_dtable ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  double **** s_ = ( N0 == 0 ) ? NULL : ( double ****) malloc( N0 * sizeof( double ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_dtable ( double ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_dtable] active\n");
    fini_3level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_dtable

/************************************************************************************/
/************************************************************************************/


inline double ***** init_5level_dtable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  double **** s__ = NULL;
  s__ = init_4level_dtable ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  double ***** s_ = ( N0 == 0 ) ? NULL : ( double *****) malloc( N0 * sizeof( double ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_dtable ( double ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_dtable] active\n");
    fini_4level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_dtable

/************************************************************************************/
/************************************************************************************/


inline double ****** init_6level_dtable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  double ***** s__ = NULL;
  s__ = init_5level_dtable ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  double ****** s_ = ( N0 == 0 ) ? NULL : ( double ******) malloc( N0 * sizeof( double *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_dtable ( double ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_dtable] active\n");
    fini_5level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_dtable

/************************************************************************************/
/************************************************************************************/


inline double ******* init_7level_dtable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  double ****** s__ = NULL;
  s__ = init_6level_dtable ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  double ******* s_ = ( N0 == 0 ) ? NULL : ( double *******) malloc( N0 * sizeof( double ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_dtable ( double ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_dtable] active\n");
    fini_6level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_dtable

/************************************************************************************/
/************************************************************************************/


inline double ******** init_8level_dtable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  double ******* s__ = NULL;
  s__ = init_7level_dtable ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  double ******** s_ = ( N0 == 0 ) ? NULL : ( double ********) malloc( N0 * sizeof( double *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_dtable

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_dtable ( double ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_dtable] active\n");
    fini_7level_dtable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_dtable

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
