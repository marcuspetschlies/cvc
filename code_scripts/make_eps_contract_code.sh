#!/bin/bash

perm_tab_3[0]=0
perm_tab_3[1]=1
perm_tab_3[2]=2
#
perm_tab_3[3]=0
perm_tab_3[4]=2
perm_tab_3[5]=1
#
perm_tab_3[6]=1
perm_tab_3[7]=2
perm_tab_3[8]=0
#
perm_tab_3[9]=1
perm_tab_3[10]=0
    perm_tab_3[11]=2
#
perm_tab_3[12]=2
perm_tab_3[13]=0
perm_tab_3[14]=1
#
perm_tab_3[15]=2
perm_tab_3[16]=1
perm_tab_3[17]=0

perm_tab_3_sign[0]=+1
perm_tab_3_sign[1]=-1
perm_tab_3_sign[2]=+1
perm_tab_3_sign[3]=-1
perm_tab_3_sign[4]=+1
perm_tab_3_sign[5]=-1

what=$1
echo "#ifndef _EPS_CONTRACT_${what}_H"
echo "#define _EPS_CONTRACT_${what}_H"
echo "#define _fp_eq_fp_eps_contract${what}_fp(_r, _u, _v) {\\"

if [ "X$what" == "X13" ]; then
# contract 1 3
for((s2=0;s2<4;s2++)); do
  for((jperm=0;jperm<6;jperm++)); do
    b3=${perm_tab_3[$((3*$jperm+0))]}
    b1=${perm_tab_3[$((3*$jperm+1))]}
    b2=${perm_tab_3[$((3*$jperm+2))]}
    j=$((3*$s2+$b3))
 
    for((s1=0;s1<4;s1++)); do
      for((iperm=0;iperm<6;iperm++)); do
        a3=${perm_tab_3[$((3*$iperm+0))]}
        a1=${perm_tab_3[$((3*$iperm+1))]}
        a2=${perm_tab_3[$((3*$iperm+2))]}
        i=$((3*$s1+$a3))

        s1b1=$((3*$s1+$b1))
        s2b2=$((3*$s2+$b2))

        if [ $((${perm_tab_3_sign[$iperm]} * ${perm_tab_3_sign[$jperm]})) -eq 1 ]; then
          sign=+
        else
          sign=-
        fi 

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          ra1=$((3*$k+$a1))
          ra2=$((3*$k+$a2))
          echo "+(_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2)))] - (_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1+1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          ra1=$((3*$k+$a1))
          ra2=$((3*$k+$a2))
          echo "+(_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2+1)))] + (_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1+1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

if [ "X$what" == "X14" ]; then
# contract 1 4
for((s2=0;s2<4;s2++)); do
  for((jperm=0;jperm<6;jperm++)); do
    b3=${perm_tab_3[$((3*$jperm+0))]}
    b1=${perm_tab_3[$((3*$jperm+1))]}
    b2=${perm_tab_3[$((3*$jperm+2))]}
    j=$((3*$s2+$b3))
 
    for((s1=0;s1<4;s1++)); do
      for((iperm=0;iperm<6;iperm++)); do
        a3=${perm_tab_3[$((3*$iperm+0))]}
        a1=${perm_tab_3[$((3*$iperm+1))]}
        a2=${perm_tab_3[$((3*$iperm+2))]}
        i=$((3*$s1+$a3))

        s1b1=$((3*$s1+$b1))
        s2a2=$((3*$s2+$a2))

        if [ $((${perm_tab_3_sign[$iperm]} * ${perm_tab_3_sign[$jperm]})) -eq 1 ]; then
          sign=+
        else
          sign=-
        fi 

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          ra1=$((3*$k+$a1))
          rb2=$((3*$k+$b2))
          echo "+(_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2)))] - (_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1+1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          ra1=$((3*$k+$a1))
          rb2=$((3*$k+$b2))
          echo "+(_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2+1)))] + (_u)[$(printf "%2d" $s1b1)][$(printf "%2d" $((2*$ra1+1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

if [ "X$what" == "X23" ]; then
# contract 2 3
for((s2=0;s2<4;s2++)); do
  for((jperm=0;jperm<6;jperm++)); do
    b3=${perm_tab_3[$((3*$jperm+0))]}
    b1=${perm_tab_3[$((3*$jperm+1))]}
    b2=${perm_tab_3[$((3*$jperm+2))]}
    j=$((3*$s2+$b3))
 
    for((s1=0;s1<4;s1++)); do
      for((iperm=0;iperm<6;iperm++)); do
        a3=${perm_tab_3[$((3*$iperm+0))]}
        a1=${perm_tab_3[$((3*$iperm+1))]}
        a2=${perm_tab_3[$((3*$iperm+2))]}
        i=$((3*$s1+$a3))

        s1a1=$((3*$s1+$a1))
        s2b2=$((3*$s2+$b2))

        if [ $((${perm_tab_3_sign[$iperm]} * ${perm_tab_3_sign[$jperm]})) -eq 1 ]; then
          sign=+
        else
          sign=-
        fi 

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          rb1=$((3*$k+$b1))
          ra2=$((3*$k+$a2))
          echo "+(_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2)))] - (_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1+1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          rb1=$((3*$k+$b1))
          ra2=$((3*$k+$a2))
          echo "+(_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2+1)))] + (_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1+1)))] * (_v)[$(printf "%2d" $s2b2)][$(printf "%2d" $((2*$ra2)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

if [ "X$what" == "X24" ]; then
# contract 2 4
for((s2=0;s2<4;s2++)); do
  for((jperm=0;jperm<6;jperm++)); do
    b3=${perm_tab_3[$((3*$jperm+0))]}
    b1=${perm_tab_3[$((3*$jperm+1))]}
    b2=${perm_tab_3[$((3*$jperm+2))]}
    j=$((3*$s2+$b3))
 
    for((s1=0;s1<4;s1++)); do
      for((iperm=0;iperm<6;iperm++)); do
        a3=${perm_tab_3[$((3*$iperm+0))]}
        a1=${perm_tab_3[$((3*$iperm+1))]}
        a2=${perm_tab_3[$((3*$iperm+2))]}
        i=$((3*$s1+$a3))

        s1a1=$((3*$s1+$a1))
        s2a2=$((3*$s2+$a2))

        if [ $((${perm_tab_3_sign[$iperm]} * ${perm_tab_3_sign[$jperm]})) -eq 1 ]; then
          sign=+
        else
          sign=-
        fi 

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          rb1=$((3*$k+$b1))
          rb2=$((3*$k+$b2))
          echo "+(_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2)))] - (_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1+1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] ${sign}= \\"
        for((k=0;k<4;k++)); do
          rb1=$((3*$k+$b1))
          rb2=$((3*$k+$b2))
          echo "+(_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2+1)))] + (_u)[$(printf "%2d" $rb1)][$(printf "%2d" $((2*$s1a1+1)))] * (_v)[$(printf "%2d" $rb2)][$(printf "%2d" $((2*$s2a2)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

echo "}"
echo "#endif"
exit 0
