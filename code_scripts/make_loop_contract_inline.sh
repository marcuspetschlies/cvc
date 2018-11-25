#######################################################################
# 
# make_loop_contract_inline.sh
#
# Do 22. Nov 16:50:30 CET 2018
#
#######################################################################
#!/bin/bash

source dimensions.in

echo "#ifndef _LOOP_CONTRACT_INLINE_H"
echo "#define _LOOP_CONTRACT_INLINE_H"

echo "#include <math.h>"
echo ""
echo "// spinor dimension: $N_SPINOR_DIM"
echo "// color dimension: $N_COLOR_DIM"
echo ""


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _contract_loop_x_spin_diluted ( double * const _r, double * const _s, double * const _p ) {"
for((s2=0; s2<$N_SPINOR_DIM; s2++)); do
  for((s1=0; s1<$N_SPINOR_DIM; s1++)); do

    mu=$(( $N_SPINOR_DIM * $s2 + $s1 ))
    mu2=$(( 2* $mu ))
    mu2p1=$(( 2* $mu + 1 ))
    echo "  (_r)[$(printf "%2d" $mu2  )] = 0.;"

    for((b=0;b<$N_COLOR_DIM;b++)); do

      k=$(($N_COLOR_DIM*$s2+$b))
      k2=$(( 2 * $k ))
      k2p1=$(( $k2 + 1 ))
      i=$(($N_COLOR_DIM*$s1+$b))
      i2=$(( 2 * $i ))
      i2p1=$(( $i2 + 1 ))

      echo "  (_r)[$(printf "%2d" $mu2  )] +=  (_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2  )] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2p1)];"
      # echo "  (_r)[$(printf "%2d" $mu2p1)] += -(_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2p1)] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2  )];"
    done  # end of loop on color index

    echo "  (_r)[$(printf "%2d" $mu2p1)] = 0.;"

    for((b=0;b<$N_COLOR_DIM;b++)); do

      k=$(($N_COLOR_DIM*$s2+$b))
      k2=$(( 2 * $k ))
      k2p1=$(( $k2 + 1 ))
      i=$(($N_COLOR_DIM*$s1+$b))
      i2=$(( 2 * $i ))
      i2p1=$(( $i2 + 1 ))

      # echo "  (_r)[$(printf "%2d" $mu2  )] +=  (_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2  )] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2p1)];"
      echo "  (_r)[$(printf "%2d" $mu2p1)] += -(_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2p1)] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2  )];"
    done  # end of loop on color index
  done  # end of loop on s1
done  # end of loop on s2
echo -e "}  /* end of _contract_loop_x_spin_diluted */ \n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _contract_loop_x_spin_color_diluted ( double * const _r, double * const _s, double * const _p ) {"
for((s2=0; s2<$N_SPINOR_DIM; s2++)); do
  for((s1=0; s1<$N_SPINOR_DIM; s1++)); do

    kappa=$(( $N_SPINOR_DIM * $s2 + $s1 ))

    for((b=0; b<$N_COLOR_DIM; b++)); do
      for((a=0; a<$N_COLOR_DIM; a++)); do

        k=$(($N_COLOR_DIM*$s2+$b))
        k2=$(( 2 * $k ))
        k2p1=$(( $k2 + 1 ))

        i=$(($N_COLOR_DIM*$s1+$a))
        i2=$(( 2 * $i ))
        i2p1=$(( $i2 + 1 ))

        lambda=$(( $N_COLOR_DIM * $b +$a ))

        mu=$(( $N_COLOR_DIM*N_COLOR_DIM * $kappa + $lambda ))
        mu2=$(( 2* $mu ))
        mu2p1=$(( 2* $mu + 1 ))

        echo "  (_r)[$(printf "%2d" $mu2  )] +=  (_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2  )] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2p1)];"
        echo "  (_r)[$(printf "%2d" $mu2p1)] += -(_p)[$(printf "%2d" $k2  )] * (_s)[$(printf "%2d" $i2p1)] + (_p)[$(printf "%2d" $k2p1)] * (_s)[$(printf "%2d" $i2  )];"

      done  # end of loop on color index a
    done  # end of loop on color index b
  done  # end of loop on s1
done  # end of loop on s2
echo -e "}  /* end of _contract_loop_x_spin_color_diluted */ \n\n"

echo "#endif"
exit 0
