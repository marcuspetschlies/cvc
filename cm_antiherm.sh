#!/bin/bash

for((i=0; i<3; i++)); do
for((k=0; k<3; k++)); do

  ik=$(( 2 * (3 * $i + $k) ))
  ik1=$(( 2 * (3 * $i + $k) + 1 ))
  ki=$(( 2 * (3 * $k + $i) ))
  ki1=$(( 2 * (3 * $k + $i) + 1 ))

  if [ $i -eq $k ]; then

cat << EOF
  (A)[$ik]  = 0.; \\
  (A)[$ik1] = (B)[$ik1]; \\
EOF
  elif [ $i -lt $k ]; then

cat << EOF
  (A)[$ik]  = 0.5*( (B)[$ik]  - (B)[$ki]  ); \\
  (A)[$ik1] = 0.5*( (B)[$ik1] + (B)[$ki1] ); \\
EOF
  else
cat << EOF
  (A)[$ik]  = -(A)[$ki]; \\
  (A)[$ik1] =  (A)[$ki1]; \\
EOF
  fi

done
done

exit 0
