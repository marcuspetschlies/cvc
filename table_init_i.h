#ifndef _TABLE_INIT_I_H
#define _TABLE_INIT_I_H

/****************************************************
 * table_init_i.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline int * init_1level_itable ( size_t const N0 ) {
  return( N0 == 0 ? NULL : ( int *) calloc ( N0 , sizeof( int ) ) );
}  // end of init_1level_itable

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_itable ( int **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_itable] active\n");
  *s = NULL;
}  // end of fini_1level_itable

/************************************************************************************/
/************************************************************************************/

inline int ** init_2level_itable (size_t const N0, size_t const N1 ) {
  int * s__ = NULL;
  s__ = init_1level_itable ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  int ** s_ = ( N0 == 0 ) ? NULL : ( int **) malloc( N0 * sizeof( int *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_itable ( int *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_itable] active\n");
    fini_1level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_itable

/************************************************************************************/
/************************************************************************************/


inline int *** init_3level_itable (size_t const N0, size_t const N1, size_t const N2 ) {
  int ** s__ = NULL;
  s__ = init_2level_itable ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  int *** s_ = ( N0 == 0 ) ? NULL : ( int ***) malloc( N0 * sizeof( int **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_itable ( int **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_itable] active\n");
    fini_2level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_itable

/************************************************************************************/
/************************************************************************************/


inline int **** init_4level_itable (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  int *** s__ = NULL;
  s__ = init_3level_itable ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  int **** s_ = ( N0 == 0 ) ? NULL : ( int ****) malloc( N0 * sizeof( int ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_itable ( int ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_itable] active\n");
    fini_3level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_itable

/************************************************************************************/
/************************************************************************************/


inline int ***** init_5level_itable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  int **** s__ = NULL;
  s__ = init_4level_itable ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  int ***** s_ = ( N0 == 0 ) ? NULL : ( int *****) malloc( N0 * sizeof( int ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_itable ( int ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_itable] active\n");
    fini_4level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_itable

/************************************************************************************/
/************************************************************************************/


inline int ****** init_6level_itable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  int ***** s__ = NULL;
  s__ = init_5level_itable ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  int ****** s_ = ( N0 == 0 ) ? NULL : ( int ******) malloc( N0 * sizeof( int *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_itable ( int ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_itable] active\n");
    fini_5level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_itable

/************************************************************************************/
/************************************************************************************/


inline int ******* init_7level_itable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  int ****** s__ = NULL;
  s__ = init_6level_itable ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  int ******* s_ = ( N0 == 0 ) ? NULL : ( int *******) malloc( N0 * sizeof( int ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_itable ( int ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_itable] active\n");
    fini_6level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_itable

/************************************************************************************/
/************************************************************************************/


inline int ******** init_8level_itable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  int ******* s__ = NULL;
  s__ = init_7level_itable ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  int ******** s_ = ( N0 == 0 ) ? NULL : ( int ********) malloc( N0 * sizeof( int *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_itable

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_itable ( int ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_itable] active\n");
    fini_7level_itable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_itable

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
