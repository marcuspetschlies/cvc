#ifndef _TABLE_INIT_C_H
#define _TABLE_INIT_C_H

/****************************************************
 * table_init_c.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

inline char * init_1level_ctable ( size_t const N0 ) {
  return( N0 == 0 ? NULL : ( char *) calloc ( N0 , sizeof( char ) ) );
}  // end of init_1level_ctable

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_ctable ( char **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_ctable] active\n");
  *s = NULL;
}  // end of fini_1level_ctable

/************************************************************************************/
/************************************************************************************/

inline char ** init_2level_ctable (size_t const N0, size_t const N1 ) {
  char * s__ = NULL;
  s__ = init_1level_ctable ( N0*N1);
  if ( s__ == NULL ) return( NULL );

  char ** s_ = ( N0 == 0 ) ? NULL : ( char **) malloc( N0 * sizeof( char *) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_2level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_2level_ctable ( char *** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_2level_ctable] active\n");
    fini_1level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_2level_ctable

/************************************************************************************/
/************************************************************************************/


inline char *** init_3level_ctable (size_t const N0, size_t const N1, size_t const N2 ) {
  char ** s__ = NULL;
  s__ = init_2level_ctable ( N0*N1, N2);
  if ( s__ == NULL ) return( NULL );

  char *** s_ = ( N0 == 0 ) ? NULL : ( char ***) malloc( N0 * sizeof( char **) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_3level_ctable ( char **** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_3level_ctable] active\n");
    fini_2level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_ctable

/************************************************************************************/
/************************************************************************************/


inline char **** init_4level_ctable (size_t const N0, size_t const N1, size_t const N2, size_t const N3 ) {
  char *** s__ = NULL;
  s__ = init_3level_ctable ( N0*N1, N2, N3);
  if ( s__ == NULL ) return( NULL );

  char **** s_ = ( N0 == 0 ) ? NULL : ( char ****) malloc( N0 * sizeof( char ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_4level_ctable ( char ***** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_4level_ctable] active\n");
    fini_3level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_ctable

/************************************************************************************/
/************************************************************************************/


inline char ***** init_5level_ctable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4 ) {
  char **** s__ = NULL;
  s__ = init_4level_ctable ( N0*N1, N2, N3, N4);
  if ( s__ == NULL ) return( NULL );

  char ***** s_ = ( N0 == 0 ) ? NULL : ( char *****) malloc( N0 * sizeof( char ****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_5level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_5level_ctable ( char ****** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_5level_ctable] active\n");
    fini_4level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_5level_ctable

/************************************************************************************/
/************************************************************************************/


inline char ****** init_6level_ctable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5 ) {
  char ***** s__ = NULL;
  s__ = init_5level_ctable ( N0*N1, N2, N3, N4, N5);
  if ( s__ == NULL ) return( NULL );

  char ****** s_ = ( N0 == 0 ) ? NULL : ( char ******) malloc( N0 * sizeof( char *****) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_6level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_6level_ctable ( char ******* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_6level_ctable] active\n");
    fini_5level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_6level_ctable

/************************************************************************************/
/************************************************************************************/


inline char ******* init_7level_ctable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6 ) {
  char ****** s__ = NULL;
  s__ = init_6level_ctable ( N0*N1, N2, N3, N4, N5, N6);
  if ( s__ == NULL ) return( NULL );

  char ******* s_ = ( N0 == 0 ) ? NULL : ( char *******) malloc( N0 * sizeof( char ******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_7level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_7level_ctable ( char ******** s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_7level_ctable] active\n");
    fini_6level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_7level_ctable

/************************************************************************************/
/************************************************************************************/


inline char ******** init_8level_ctable (size_t const N0, size_t const N1, size_t const N2, size_t const N3, size_t const N4, size_t const N5, size_t const N6, size_t const N7 ) {
  char ******* s__ = NULL;
  s__ = init_7level_ctable ( N0*N1, N2, N3, N4, N5, N6, N7);
  if ( s__ == NULL ) return( NULL );

  char ******** s_ = ( N0 == 0 ) ? NULL : ( char ********) malloc( N0 * sizeof( char *******) );
  if ( s_ == NULL ) return ( NULL );

  for ( size_t i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_8level_ctable

/************************************************************************************/
/************************************************************************************/


inline void fini_8level_ctable ( char ********* s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_8level_ctable] active\n");
    fini_7level_ctable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_8level_ctable

/************************************************************************************/
/************************************************************************************/


}  /* end of namespace cvc */

#endif
