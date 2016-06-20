#!/bin/bash
ofile="define_pack_matrix"
echo "/* `date` */" > $ofile
echo "/* pack 6x6 matrix in _Complex double from three 3x3 in cvc complex */" >> $ofile
echo "#define _PACK_MATRIX(a_,b_,c_,d_) {\\" >> $ofile
for((i=0; i<6; i++)); do
for((k=0; k<6; k++)); do

  ik=$((6 * $i + $k))

  if [ $i -lt 3 ]; then
    if [ $k -lt 3 ]; then
      # upper left
      i1=$i
      k1=$k
      i1k1=$((3 * $i1 + $k1))
      echo "  (a_)[`printf "%2d" $ik`] = (b_)[`printf "%2d" $((2*$i1k1))`] + (b_)[`printf "%2d" $((2*$i1k1+1))`] * I; \\"
    else
      # upper right
      i1=$i
      k1=$(($k-3))
      i1k1=$((3 * $i1 + $k1))
      echo "  (a_)[`printf "%2d" $ik`] = (c_)[`printf "%2d" $((2*$i1k1))`] + (c_)[`printf "%2d" $((2*$i1k1+1))`] * I; \\"
    fi
  else
    if [ $k -lt 3 ]; then
      # lower left
      i1=$(($i-3))
      k1=$k
      i1k1=$((3 * $k1 + $i1))
      echo "  (a_)[`printf "%2d" $ik`] = (c_)[`printf "%2d" $((2*$i1k1))`] - (c_)[`printf "%2d" $((2*$i1k1+1))`] * I; \\"
    else
      # lower right
      i1=$(($i-3))
      k1=$(($k-3))
      i1k1=$((3 * $i1 + $k1))
      echo "  (a_)[`printf "%2d" $ik`] = (d_)[`printf "%2d" $((2*$i1k1))`] + (d_)[`printf "%2d" $((2*$i1k1+1))`] * I; \\"
    fi
  fi

done
done  >> $ofile
echo "}" >> $ofile

ofile="define_unpack_matrix"
echo "/* `date` */" > $ofile
echo "/* unpack 6x6 matrix in _Complex double to four 3x3 in cvc complex */" >> $ofile
echo "#define _UNPACK_MATRIX(a_,b_,c_,d_,e_) {\\" >> $ofile

# upper left
for((i1=0; i1<3; i1++)); do
for((k1=0; k1<3; k1++)); do
  i1k1=$((3 * $i1 + $k1))
  i=$i1
  k=$k1
  ik=$((6 * $i + $k))
  echo "  (b_)[`printf "%2d" $((2*$i1k1))`] = creal( (a_)[`printf "%2d" $ik`] ); "\
       "(b_)[`printf "%2d" $((2*$i1k1+1))`] = cimag( (a_)[`printf "%2d" $ik`] ); \\"
done
done  >> $ofile

# upper right
for((i1=0; i1<3; i1++)); do
for((k1=0; k1<3; k1++)); do
  i1k1=$((3 * $i1 + $k1))
  i=$i1
  k=$(($k1+3))
  ik=$((6 * $i + $k))
  echo "  (c_)[`printf "%2d" $((2*$i1k1))`] = creal( (a_)[`printf "%2d" $ik`] ); "\
  "(c_)[`printf "%2d" $((2*$i1k1+1))`] = cimag( (a_)[`printf "%2d" $ik`] ); \\"
done
done  >> $ofile

# lower left
for((i1=0; i1<3; i1++)); do
for((k1=0; k1<3; k1++)); do
  i1k1=$((3 * $i1 + $k1))
  i=$(($i1+3))
  k=$k1
  ik=$((6 * $i + $k))
  echo "  (d_)[`printf "%2d" $((2*$i1k1))`] = creal( (a_)[`printf "%2d" $ik`] ); "\
  "(d_)[`printf "%2d" $((2*$i1k1+1))`] = cimag( (a_)[`printf "%2d" $ik`] ); \\"
done
done  >> $ofile

# lower right
for((i1=0; i1<3; i1++)); do
for((k1=0; k1<3; k1++)); do
  i1k1=$((3 * $i1 + $k1))
  i=$(($i1+3))
  k=$(($k1+3))
  ik=$((6 * $i + $k))
  echo "  (e_)[`printf "%2d" $((2*$i1k1))`] = creal( (a_)[`printf "%2d" $ik`] ); "\
  "(e_)[`printf "%2d" $((2*$i1k1+1))`] = cimag( (a_)[`printf "%2d" $ik`] ); \\"
done
done  >> $ofile

echo "}" >> $ofile

exit 0
