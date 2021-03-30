#!/bin/bash
MyName=$(echo $0 | awk -F\/ '{sub(/\.sh$/,"",$NF);print $NF}')
log=$MyName.log
out=$MyName.out
err=$MyName.err

echo "# [$MyName] `date`"

topLevelDir=$PWD

input=cvc.input
out=out
err=err
log=log

export OMP_NUM_THREADS=4

exec_dir=

# valgrind -v --leak-check=full --show-reachable=yes \

${exec_dir}/njjn_fht_invert_contract -c  -f $input 1>$out 2>$err

es=$?
if [ $es -ne 0 ]; then
  echo "[$MyName] Error from prog, status was $es"
  exit 1
else
  echo "# [$MyName] status 0"
fi

echo "# [$MyName] finished all"
echo "# [$MyName] `date`"
exit 0
