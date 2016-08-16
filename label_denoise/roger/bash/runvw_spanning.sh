#!/bin/bash

onlinepasses=$1
batchpasses=$2
regularization=$3
nbits=$4
quad_flags=$5

IFS=',' read -ra quadcomb <<< "$quad_flags"
vwquadcomb=""
for q in "${quadcomb[@]}"; do      
	vwquadcomb=$vwquadcomb" -q "$q
done

mapreduce_job_id=`echo $mapreduce_job_id | tr -d 'job_'`
./vw -b $nbits \
	--total $mapreduce_job_maps \
	--node $mapreduce_task_partition \
	--unique_id $mapreduce_job_id \
	--save_per_pass $vwquadcomb \
	--adaptive \
	--exact_adaptive_norm \
	--cache_file temp.cache \
	--passes $onlinepasses \
	-d /dev/stdin \
	-f model_ \
	--span_server $mapreduce_job_submithostname \
	--loss_function=logistic 2>&1 | tee output_online >/dev/stderr

if [ ${PIPESTATUS[0]} -ne 0 ]
then
   exit 1
fi

#use bc because of numerical issue with exp
mapreduce_job_id=`echo $mapreduce_job_id \* 2 | bc`
./vw --total $mapreduce_job_maps --node $mapreduce_task_partition --unique_id $mapreduce_job_id --save_per_pass --cache_file temp.cache -b $nbits $vwquadcomb --passes $batchpasses --regularization=$regularization --span_server $mapreduce_job_submithostname -d /dev/stdin -f model -i model_ --bfgs --mem 10 --loss_function=logistic 2>&1 | tee output_bfgs >/dev/stderr

if [ "$mapreduce_task_partition" == '0' ]
then
    tar -zcvf model.tar.gz model* >/dev/stderr
    $HADOOP_PREFIX/bin/hadoop fs -rm -r $mapreduce_output_fileoutputformat_outputdir/model.tar.gz
    $HADOOP_PREFIX/bin/hadoop fs -put model.tar.gz output_* $mapreduce_output_fileoutputformat_outputdir
fi

