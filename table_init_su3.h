#ifndef _TABLE_INIT_SU3_H
#define _TABLE_INIT_SU3_H

/****************************************************
 * table_init_su3.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline su3 * init_1level_su3table ( size_t const N0 ) {
  fprintf( stdout, "# [init_1level_su3table] sizeof su3 = %lu\n", sizeof(su3) );
  fflush( stdout );
  return( N0 == 0 ? NULL : ( su3 *) calloc ( N0 , sizeof( su3 ) ) );
}  // end of init_1level_su3table

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_su3table ( su3 **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_su3table] active\n");
  *s = NULL;
}  // end of fini_1level_su3table

/************************************************************************************/
/************************************************************************************/

inline su3 ** init_2level_su3table (size_t const N0, size_t const N1 ) {
  su3 * s__ = NULL;
  s__ = init_1level_su3table ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  su3 ** s_ = ( N0 == 0 ) ? NULL : ( su3 **) malloc( N0 * sizeof( su3 *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_su3table ( su3 *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_su3table] active\n");
    fini_1level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 *** init_3level_su3table (size_t const N0, size_t const N1, size_t const N2 ) {
  su3 ** s__ = NULL;
  s__ = init_2level_su3table ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  su3 *** s_ = ( N0 == 0 ) ? NULL : ( su3 ***) malloc( N0 * sizeof( su3 **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_su3table ( su3 **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_su3table] active\n");
    fini_2level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 **** init_4level_su3table (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  su3 *** s__ = NULL;
  s__ = init_3level_su3table ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  su3 **** s_ = ( N0 == 0 ) ? NULL : ( su3 ****) malloc( N0 * sizeof( su3 ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_su3table ( su3 ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_su3table] active\n");
    fini_3level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 ***** init_5level_su3table (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  su3 **** s__ = NULL;
  s__ = init_4level_su3table ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  su3 ***** s_ = ( N0 == 0 ) ? NULL : ( su3 *****) malloc( N0 * sizeof( su3 ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_su3table ( su3 ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_su3table] active\n");
    fini_4level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 ****** init_6level_su3table (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  su3 ***** s__ = NULL;
  s__ = init_5level_su3table ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  su3 ****** s_ = ( N0 == 0 ) ? NULL : ( su3 ******) malloc( N0 * sizeof( su3 *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_su3table ( su3 ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_su3table] active\n");
    fini_5level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 ******* init_7level_su3table (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  su3 ****** s__ = NULL;
  s__ = init_6level_su3table ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  su3 ******* s_ = ( N0 == 0 ) ? NULL : ( su3 *******) malloc( N0 * sizeof( su3 ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_su3table ( su3 ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_su3table] active\n");
    fini_6level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_su3table

/************************************************************************************/
/************************************************************************************/


inline su3 ******** init_8level_su3table (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  su3 ******* s__ = NULL;
  s__ = init_7level_su3table ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  su3 ******** s_ = ( N0 == 0 ) ? NULL : ( su3 ********) malloc( N0 * sizeof( su3 *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_su3table

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_su3table ( su3 ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_su3table] active\n");
    fini_7level_su3table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_su3table

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
