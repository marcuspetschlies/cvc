#ifndef _TABLE_INIT_F_H
#define _TABLE_INIT_F_H

/****************************************************
 * table_init_f.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline float * init_1level_ftable ( size_t const N0 ) {
  return( N0 == 0 ? NULL : ( float *) calloc ( N0 , sizeof( float ) ) );
}  // end of init_1level_ftable

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_ftable ( float **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_ftable] active\n");
  *s = NULL;
}  // end of fini_1level_ftable

/************************************************************************************/
/************************************************************************************/

inline float ** init_2level_ftable (size_t const N0, size_t const N1 ) {
  float * s__ = NULL;
  s__ = init_1level_ftable ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  float ** s_ = ( N0 == 0 ) ? NULL : ( float **) malloc( N0 * sizeof( float *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_ftable ( float *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_ftable] active\n");
    fini_1level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_ftable

/************************************************************************************/
/************************************************************************************/


inline float *** init_3level_ftable (size_t const N0, size_t const N1, size_t const N2 ) {
  float ** s__ = NULL;
  s__ = init_2level_ftable ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  float *** s_ = ( N0 == 0 ) ? NULL : ( float ***) malloc( N0 * sizeof( float **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_ftable ( float **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_ftable] active\n");
    fini_2level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_ftable

/************************************************************************************/
/************************************************************************************/


inline float **** init_4level_ftable (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  float *** s__ = NULL;
  s__ = init_3level_ftable ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  float **** s_ = ( N0 == 0 ) ? NULL : ( float ****) malloc( N0 * sizeof( float ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_ftable ( float ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_ftable] active\n");
    fini_3level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_ftable

/************************************************************************************/
/************************************************************************************/


inline float ***** init_5level_ftable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  float **** s__ = NULL;
  s__ = init_4level_ftable ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  float ***** s_ = ( N0 == 0 ) ? NULL : ( float *****) malloc( N0 * sizeof( float ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_ftable ( float ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_ftable] active\n");
    fini_4level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_ftable

/************************************************************************************/
/************************************************************************************/


inline float ****** init_6level_ftable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  float ***** s__ = NULL;
  s__ = init_5level_ftable ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  float ****** s_ = ( N0 == 0 ) ? NULL : ( float ******) malloc( N0 * sizeof( float *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_ftable ( float ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_ftable] active\n");
    fini_5level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_ftable

/************************************************************************************/
/************************************************************************************/


inline float ******* init_7level_ftable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  float ****** s__ = NULL;
  s__ = init_6level_ftable ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  float ******* s_ = ( N0 == 0 ) ? NULL : ( float *******) malloc( N0 * sizeof( float ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_ftable ( float ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_ftable] active\n");
    fini_6level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_ftable

/************************************************************************************/
/************************************************************************************/


inline float ******** init_8level_ftable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  float ******* s__ = NULL;
  s__ = init_7level_ftable ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  float ******** s_ = ( N0 == 0 ) ? NULL : ( float ********) malloc( N0 * sizeof( float *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_ftable

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_ftable ( float ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_ftable] active\n");
    fini_7level_ftable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_ftable

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
