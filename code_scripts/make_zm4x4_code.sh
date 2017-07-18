#!/bin/bash

if [ 1 -eq 0 ]; then
for i in 0 1 2 3; do
  for k in 0 1 2 3; do
    ik=$((4*$i+$k))
    ki=$((4*$k+$i))
    printf "  r[%2d] = s[%2d];\n" $ik $ki
  done
done
fi

if [ 1 -eq 0 ]; then
for i in 0 1 2 3; do
  for k in 0 1 2 3; do
    ik=$(( 4*$i + $k ))
    l=0
    il=$(( 4*$i + $l ))
    lk=$(( 4*$l + $k ))
    printf "  r[0][%2d] = ss[%2d] * tt[%2d]" $ik  $il $lk
    for(( l=1; l<=2; l++)); do
      il=$(( 4*$i + $l ))
      lk=$(( 4*$l + $k ))
      printf " + ss[%2d] * tt[%2d]" $il $lk
    done
    il=$(( 4*$i + $l ))
    lk=$(( 4*$l + $k ))
    printf " + ss[%2d] * tt[%2d];\n" $il $lk
  done
done
fi

# spin-parity projection 1/2 x ( 1 + c gamma_0 ), gamma_0 in tmLQCD basis
if [ 1 -eq 0 ]; then
for i1 in 0 1 2 3; do
  k1=$(( ($i1+2)%4 ))
  for i2 in 0 1 2 3; do
    k2=$i2
    ii=$((4*$i1+$i2))
    kk=$((4*$k1+$k2))
    printf "  r[0][%2d] = ( b[%2d] - b[%2d] * c ) * norm;\n" $ii $ii $kk
  done
done
fi

# r = s x t + c u
if [ 1 -eq 0 ]; then
for i in 0 1 2 3; do
  for k in 0 1 2 3; do
    ik=$(( 4*$i + $k ))
    l=0
    il=$(( 4*$i + $l ))
    lk=$(( 4*$l + $k ))
    printf "  r[0][%2d] = ss[%2d] * tt[%2d] " $ik  $il $lk
    for(( l=1; l<=2; l++)); do
      il=$(( 4*$i + $l ))
      lk=$(( 4*$l + $k ))
      printf " + ss[%2d] * tt[%2d] " $il $lk
    done
    il=$(( 4*$i + $l ))
    lk=$(( 4*$l + $k ))
    printf " + ss[%2d] * tt[%2d] + u[0][%2d] * c;\n" $il $lk $ik
  done
done
fi

# c = r - s; p = tr ( p^+ p )
if [ 1 -eq 0 ]; then
printf "  pp = 0.;\n"
for i in 0 1 2 3; do
  for k in 0 1 2 3; do
    ik=$(( 4*$i + $k ))
    printf "  c   = r[0][%2d] - s[0][%2d];" $ik $ik
    printf "  pp += creal( c * conj(c) );\n"
  done
done
printf "  (*p) = sqrt(pp);\n"
fi

# r = s x c, r, s zm4x4 , c double _Complex
#if [ 1 -eq 0 ]; then
  for i in 0 1 2 3; do
    for k in 0 1 2 3; do
      ik=$(( 4*$i + $k ))
      printf "  r[0][%2d] = s[0][%2d] * c;\n" $ik  $ik
    done
  done
#fi

exit 0
