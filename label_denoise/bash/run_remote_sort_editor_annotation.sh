#!/bin/sh
#source
source ~/.bash_profile
cd ~/src
# ticket
kinit <<< PASSWORD@

nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com
src_path="${HOME}/label_denoise/";
infile="${nn}/user/wangxl/mt.out";
outfile=${infile/"out"/"out.sort"};
cd ${src_path};
hadoop fs -cat $infile | zcat | python test.py --sort_mt_annotation > tmp;
hadoop fs -copyFromLocal tmp;
rm -rf tmp;
