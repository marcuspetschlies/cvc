#ifndef _TABLE_INIT_2PT_H
#define _TABLE_INIT_2PT_H

/****************************************************
 * table_init_2pt.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline twopoint_function_type * init_1level_2pttable ( size_t const N0 ) {
  return( N0 == 0 ? NULL : ( twopoint_function_type *) calloc ( N0 , sizeof( twopoint_function_type ) ) );
}  // end of init_1level_2pttable

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_2pttable ( twopoint_function_type **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_2pttable] active\n");
  *s = NULL;
}  // end of fini_1level_2pttable

/************************************************************************************/
/************************************************************************************/

inline twopoint_function_type ** init_2level_2pttable (size_t const N0, size_t const N1 ) {
  twopoint_function_type * s__ = NULL;
  s__ = init_1level_2pttable ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type ** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type **) malloc( N0 * sizeof( twopoint_function_type *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_2pttable ( twopoint_function_type *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_2pttable] active\n");
    fini_1level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type *** init_3level_2pttable (size_t const N0, size_t const N1, size_t const N2 ) {
  twopoint_function_type ** s__ = NULL;
  s__ = init_2level_2pttable ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type *** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type ***) malloc( N0 * sizeof( twopoint_function_type **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_2pttable ( twopoint_function_type **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_2pttable] active\n");
    fini_2level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type **** init_4level_2pttable (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  twopoint_function_type *** s__ = NULL;
  s__ = init_3level_2pttable ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type **** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type ****) malloc( N0 * sizeof( twopoint_function_type ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_2pttable ( twopoint_function_type ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_2pttable] active\n");
    fini_3level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type ***** init_5level_2pttable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  twopoint_function_type **** s__ = NULL;
  s__ = init_4level_2pttable ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type ***** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type *****) malloc( N0 * sizeof( twopoint_function_type ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_2pttable ( twopoint_function_type ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_2pttable] active\n");
    fini_4level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type ****** init_6level_2pttable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  twopoint_function_type ***** s__ = NULL;
  s__ = init_5level_2pttable ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type ****** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type ******) malloc( N0 * sizeof( twopoint_function_type *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_2pttable ( twopoint_function_type ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_2pttable] active\n");
    fini_5level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type ******* init_7level_2pttable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  twopoint_function_type ****** s__ = NULL;
  s__ = init_6level_2pttable ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type ******* s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type *******) malloc( N0 * sizeof( twopoint_function_type ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_2pttable ( twopoint_function_type ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_2pttable] active\n");
    fini_6level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_2pttable

/************************************************************************************/
/************************************************************************************/


inline twopoint_function_type ******** init_8level_2pttable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  twopoint_function_type ******* s__ = NULL;
  s__ = init_7level_2pttable ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  twopoint_function_type ******** s_ = ( N0 == 0 ) ? NULL : ( twopoint_function_type ********) malloc( N0 * sizeof( twopoint_function_type *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_2pttable

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_2pttable ( twopoint_function_type ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_2pttable] active\n");
    fini_7level_2pttable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_2pttable

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
