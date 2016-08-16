            DEDFSDFSDFSA#!/bin/bash

train_directory=$1
test_directory=$2
onlinepasses=$3
batchpasses=$4
regularization=$5
nbits=$6
modeltag=$7
model_directory=$8
quad_comb=$9

tmp_dir=$USER-$RANDOM
model_directory=$model_directory/${modeltag}_nbits_${nbits}_reg_${regularization}
output_directory=$model_directory/pred

CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON="epd/bin/python"
nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com:8020
#nn=hftp://nitroblue-nn1.blue.ygrid.yahoo.com:50070
vwpath=/homes/luojie/src/vowpal_wabbit

#hadoop fs -rm -r $model_directory
$vwpath/cluster/spanning_tree

train_mapper="runvw_spanning.sh $onlinepasses $batchpasses $regularization $nbits $quad_comb" 

train_directory=`echo $train_directory | sed "s/;/ /g"`
input_train_dir=""
for itd in $train_directory
do
	input_train_dir=${input_train_dir}" -input "${itd}
done
echo "Splitted inputs: $input_train_dir"

# Map-only job that uses the spanning tree for (1) an online warm-start pass (2) parallel execution of l-BFGS
hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar \
			-Dmapred.map.tasks.speculative.execution=true \
			-Dmapred.job.queue.name=search_fast_lane \
			-Dmapred.reduce.tasks=0 \
			-Dmapred.job.map.memory.mb=3072 \
			-Dmapred.child.java.opts="-Xmx1000m" \
			-Dmapred.task.timeout=600000000 \
			-files $vwpath/vw,runvw_spanning.sh \
			$input_train_dir -output $model_directory -mapper "$train_mapper" -reducer NONE

# clean-up
hadoop fs -rm -r -skipTrash $model_directory/part-*

# one per model file, essentially
nreducers=`expr $batchpasses + $onlinepasses`

# The mappers compute the predictions of the model: (k,v) where the key is the model file tag and the value is the prediction
# The reducers group/split the predictions into one file per model tag, then apply the policy evaluator on each file.
hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar \
			-Dmapred.child.env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./epd/lib" \
			-Dmapred.cache.archives=$CACHE_PYTHON \
			-Dmapred.map.tasks.speculative.execution=true \
			-Dmapred.reduce.speculative.execution=true \
			-Dmapred.job.queue.name=search_fast_lane \
			-Dmapred.reduce.tasks=$nreducers \
			-Dmapred.job.reduce.memory.mb=4000 \
			-Dmapred.job.map.memory.mb=3000 \
			-Dmapred.child.java.opts="-Xmx512m" \ 
			-Dmapred.task.timeout=600000000 \
			-files $nn$model_directory/model.tar.gz,$vwpath/vw,testvw.sh,vwPolicyEvaluator_scipy.py,testvw_reducer.py \
			-input $test_directory \
			-output $output_directory \
			-mapper testvw.sh \
			-reducer "python testvw_reducer.py $output_directory eval"

# clean-up
hadoop fs -rm -r -skipTrash $output_directory/part-*

hadoop archive -Dmapred.output.compress=true -Dmapred.job.queue.name=search_fast_lane -archiveName preds_and_eval.har -p $output_directory $model_directory

#hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar -Dmapred.job.queue.name=search_fast_lane -Dmapred.map.tasks=1 -files $nn$output_directory -output ${output_directory}_tar -input $output_directory/1.pred -file tar_mapper.sh -mapper tar_mapper.sh -reducer NONE

if [ "$?" = "0" ]; then
	# clean-up
	hadoop fs -rm -r -skipTrash $output_directory/*pred
	hadoop fs -rm -r -skipTrash $output_directory/*eval
else
	echo "Something went wrong with the tar mapreduce job"
	echo "Keeping the files intact in " $output_directory
fi
