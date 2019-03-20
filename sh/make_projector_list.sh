#!/bin/bash
MyName=$(echo $0 | awk -F\/ '{sub(/\.sh$/,"",$NF);print $NF}')
log=$MyName.log
out=$MyName.out
err=$MyName.err


sigma_g0d=( 1 -1 -1 -1  1 -1  1 -1 -1 -1  1  1  1 -1 -1 -1 )

########################################
# get irrep dimension from first
#   character of irrep name
########################################
function get_irrep_dim {
  if   [[ $1 == H* ]]; then echo 4;
  elif [[ $1 == G* ]]; then echo 2;
  elif [[ $1 == K* ]]; then echo 1;
  elif [[ $1 == A* ]]; then echo 1;
  elif [[ $1 == B* ]]; then echo 1;
  elif [[ $1 == E* ]]; then echo 2;
  elif [[ $1 == T* ]]; then echo 3;
  else
    echo "[get_irrep_dim] Error, unknown irrep initial"
    exit 2
  fi
}


echo "# [$MyName] (`date`)" > $log

tag=$1

if [ "X$tag" == "X" ] ; then
  echo "[$MyName] Error, need tag"
  exit 1
fi >> $log

########################################
# number of sink timeslices
#  ( not including source time )
########################################
src_snk_time_separation=24

cat << EOF >> $log
# [$MyName] tag             = $tag
# [$MyName] src_snk_time_separation = $src_snk_time_separation
EOF

########################################
# list of little groups
########################################
littlegroup_list=( 2Oh 2C4v 2C2v 2C3v )

########################################
# total momentum list per little group
########################################
ptot_list[0]="0,0,0"
ptot_list[1]="0,0,1;0,0,-1; 1,0,0;-1,0,0;0,1,0;0,-1,0;"
ptot_list[2]="1,1,0;-1,-1,0;1,0,1;-1,0,-1;0,1,1;0,-1,-1;-1,1,0;1,-1,0;1,0,-1;-1,0,1;0,-1,1;0,1,-1"
ptot_list[3]="1,1,1;-1,-1,-1;1,1,-1;-1,-1,1;1,-1,1;-1,1,-1;-1,1,1;1,-1,-1"

########################################
if [ "X$tag" == "XN-N" ]; then
########################################

  type="b-b"

  irrep_list[0]="G1g;G2g;G1u;G2u"
  irrep_list[1]="G1;G2"
  irrep_list[2]="G1"
  irrep_list[3]="K1;K2;G1"

  gi1_list=(  14,4  11,5  8,4 )
  gf1_list=(  14,4  11,5  8,4 )

  fbwd_list=( fwd bwd )

  d_spin=4
  reorder=0
  num_diag=2
  diagrams="n1,n2"

cat << EOF

BeginTwopointFunctionGeneric
  n        = $num_diag
  d        = $d_spin
  type     = $type
  tag      = $tag
  reorder  = $reorder
  T        = $src_snk_time_separation
  diagrams = $diagrams
EndTwopointFunction

EOF

for((ilg=0; ilg<4; ilg++)); do
  lg=${littlegroup_list[$ilg]}

  nptot=$( echo ${ptot_list[$ilg]} | tr ';' ' ' | wc -w )
  echo "# [$MyName] lg = $lg nptot = $nptot" >> $log

  ### for(( iptot=0; iptot<$nptot; iptot++ ));
  for(( iptot=0; iptot<1; iptot++ ));
  do

    ptot=$( echo ${ptot_list[$ilg]} | awk -F\; '{print $('$iptot'+1)}' )

    pf1=$ptot
    pi1=$(echo $ptot | tr ',' ' ' | awk '{printf("%d,%d,%d", -$1, -$2, -$3)}')
 
    echo "# [$MyName] lg = $lg ptot = $ptot pi1 = $pi1 pf1 = $pf1" >> $log

    nirrep=$(echo ${irrep_list[$ilg]} | tr ';' ' ' | wc -w)

    for(( iirrep=0; iirrep<$nirrep; iirrep++)) ; do

      irrep=$( echo ${irrep_list[$ilg]} | awk -F\; '{print $('$iirrep'+1)}' )

      irrep_dim=$( get_irrep_dim $irrep )

      echo "# [$MyName] lg = $lg ptot = $ptot irrep = $irrep irrep_dim = $irrep_dim" >> $log

      for fbwd in "fwd" "bwd"; do

        for gi1 in ${gi1_list[*]}; do
        for gf1 in ${gf1_list[*]}; do

cat << EOF

BeginTwopointFunctionInit
  irrep    = $irrep
  pi1      = $pi1
  pf1      = $pf1
  gi1      = $gi1
  gf1      = $gf1
  group    = $lg
  fbwd     = $fbwd
EndTwopointFunction

EOF
        done  # of gf1
        done  # of gi1
      done  # of fbwd
    done  # of irrep
  done  # of ptot
done  # of ilg

########################################
elif [ "X$tag" == "XD-D" ]; then
########################################

  type="b-b"

  irrep_list[0]="G1g;G2g;Hg;G1u;G2u;Hu"
  irrep_list[1]="G1;G2"
  irrep_list[2]="G1"
  irrep_list[3]="K1;K2;G1"


  gi1_list=(  9,4  0,4  7,4 13,4  4,4 15,4 )
  gf1_list=(  9,4  0,4  7,4 13,4  4,4 15,4 )

  fbwd_list=( fwd bwd )

  d_spin=4
  reorder=0
  diagrams="d1,d2,d3,d4,d5,d6"
  num_diag=$(echo $diagrams | tr ',' ' ' | wc -w )
  echo "# [$MyName] diagrams = $diagrams, num_diag = $num_diag" >> $log

cat << EOF

BeginTwopointFunctionGeneric
  n        = $num_diag
  d        = $d_spin
  type     = $type
  tag      = $tag
  reorder  = $reorder
  T        = $src_snk_time_separation
  diagrams = $diagrams
EndTwopointFunction

EOF

for((ilg=0; ilg<4; ilg++)); do
  lg=${littlegroup_list[$ilg]}

  nptot=$( echo ${ptot_list[$ilg]} | tr ';' ' ' | wc -w )
  echo "# [$MyName] lg = $lg nptot = $nptot" >> $log

  ### for(( iptot=0; iptot<$nptot; iptot++ ));
  for(( iptot=0; iptot<1; iptot++ ));
  do

    ptot=$( echo ${ptot_list[$ilg]} | awk -F\; '{print $('$iptot'+1)}' )

    pf1=$ptot
    pi1=$(echo $ptot | tr ',' ' ' | awk '{printf("%d,%d,%d", -$1, -$2, -$3)}')
 
    echo "# [$MyName] lg = $lg ptot = $ptot pi1 = $pi1 pf1 = $pf1" >> $log

    nirrep=$(echo ${irrep_list[$ilg]} | tr ';' ' ' | wc -w)

    for(( iirrep=0; iirrep<$nirrep; iirrep++)) ; do

      irrep=$( echo ${irrep_list[$ilg]} | awk -F\; '{print $('$iirrep'+1)}' )

      irrep_dim=$( get_irrep_dim $irrep )

      echo "# [$MyName] lg = $lg ptot = $ptot irrep = $irrep irrep_dim = $irrep_dim" >> $log

      for fbwd in "fwd" "bwd"; do

        for gi1 in ${gi1_list[*]}; do
          
          gi1_12=($( echo $gi1 | tr ',' ' ' ))

          norm_str="  # norm     ="
          s0d=${sigma_g0d[${gi1_12[0]}]}
          if [ $s0d -ne 1 ]; then
            norm_str="  norm     = $s0d"
            for((i=1;i<$num_diag -1 ;i++));do norm_str="${norm_str},${sigma_g0d[${gi1_12[0]}]}"; done
          fi

        for gf1 in ${gf1_list[*]}; do

          gf1_12=($( echo $gf1 | tr ',' ' ' ))

cat << EOF

BeginTwopointFunctionInit
  irrep    = $irrep
  pi1      = $pi1
  pf1      = $pf1
  gi1      = $gi1
  gf1      = $gf1
  group    = $lg
  fbwd     = $fbwd
$norm_str
EndTwopointFunction

EOF
        done  # of gf1
        done  # of gi1
      done  # of fbwd
    done  # of irrep
  done  # of ptot
done  # of ilg

fi

echo "# [$MyName] (`date`)" >> $log
exit 0
