#!/bin/bash
MyName=$(echo $0 | awk -F\/ '{sub(/\.sh/,"",$NF);print $NF}')

#TYPE="double _Complex"
#TAG="z"

#TYPE="double"
#TAG="d"

#TYPE="int"
#TAG="i"

#TYPE="char"
#TAG="c"

#TYPE="twopoint_function_type"
#TAG="2pt"

TAG=$1
if [ "X$TAG" == "X" ]; then
  echo "[$MyName] Error, need type identifier"
  exit 1
fi

case "$TAG" in
  "z") TYPE="double _Complex";;
  "d") TYPE="double";;
  "i") TYPE="int";;
  "c") TYPE="char";;
  "2pt") TYPE="twopoint_function_type";;
  *) exit 1;;
esac

TTAG=$( echo $TAG | tr '[:lower:]' '[:upper:]')

SIZE_TYPE="size_t"

FILE=table_init_${TAG}.h

cat << EOF > $FILE
#ifndef _TABLE_INIT_${TTAG}_H
#define _TABLE_INIT_${TTAG}_H

/****************************************************
 * table_init_${TAG}.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {
EOF


cat << EOF >> $FILE

inline $TYPE * init_1level_${TAG}table ( ${SIZE_TYPE} const N0 ) {
  return( N0 == 0 ? NULL : ( $TYPE *) calloc ( N0 , sizeof( $TYPE ) ) );
}  // end of init_1level_${TAG}table

/************************************************************************************/
/************************************************************************************/

inline void fini_1level_${TAG}table ( ${TYPE} **s  ) {
  if ( *s != NULL ) free ( *s );
  // fprintf ( stdout, "# [fini_1level_${TAG}table] active\n");
  *s = NULL;
}  // end of fini_1level_${TAG}table

/************************************************************************************/
/************************************************************************************/

EOF



for LEVEL in $(seq 2 8 ); do

  PTR2=""
  for ((k=1; k < $LEVEL; k++ )) do
    PTR2="${PTR2}*"
  done
  PTR="${PTR2}*"
  PTR3="${PTR}*"

  printf "inline %s %s init_%dlevel_%stable (" "$TYPE" "$PTR"  $LEVEL "$TAG"
for ((k=1; k < $LEVEL; k++ )) do
  printf "${SIZE_TYPE} const N%d, " $(($k - 1 ))
done
printf "${SIZE_TYPE} const N%d ) {\n" $(($LEVEL - 1 ))

printf "  %s %s s__ = NULL;\n" "${TYPE}" "$PTR2"
printf "  s__ = init_%dlevel_%stable ( N0*N1" $(( $LEVEL - 1 )) "$TAG"

for ((k=2; k < $LEVEL; k++ )) do
  printf ", N%d" $k
done
printf ");\n"

cat << EOF
  if ( s__ == NULL ) return( NULL );

  ${TYPE} $PTR s_ = ( N0 == 0 ) ? NULL : ( ${TYPE} $PTR) malloc( N0 * sizeof( ${TYPE} $PTR2) );
  if ( s_ == NULL ) return ( NULL );

  for ( ${SIZE_TYPE} i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_${LEVEL}level_${TAG}table

/************************************************************************************/
/************************************************************************************/


inline void fini_${LEVEL}level_${TAG}table ( ${TYPE} ${PTR3} s  ) {
  if ( *s != NULL ) {
    // fprintf ( stdout, "# [fini_${LEVEL}level_${TAG}table] active\n");
    fini_$(( $LEVEL - 1))level_${TAG}table ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_${LEVEL}level_${TAG}table

/************************************************************************************/
/************************************************************************************/


EOF

done >> $FILE

cat << EOF >> $FILE
}  /* end of namespace cvc */

#endif
EOF

exit 0
