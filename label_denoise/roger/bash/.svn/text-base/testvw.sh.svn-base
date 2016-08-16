#!/bin/bash
tar -zxvf model.tar.gz >& /dev/null
rm -f model.tar.gz
mv model model.final
for file in `ls -1 model*`
do
  model_number=`echo $file | tr -d '[model._]'`
  echo _______ $model_number >/dev/stderr
  ./vw -t -d /dev/stdin --loss_function=logistic -i $file --cache_file temp.cache -p /dev/stdout | awk -v OFS='\t' -v mn=$model_number '{print mn,$1,$2}'
done

