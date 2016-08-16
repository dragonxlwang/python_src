#! /bin/sh

svn update

# hadoop params
CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON="epd/bin/python"
nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com


# clean
hadoop fs -rm -r $nn"/user/wangxl/test.out"

# run

trainDayLst="20121129 20121130 20121201 20121202 20121203 20121204 20121204 20121205 20121206 20121207 20121208 20121209 20121210 20121211 20121213";
trainDayLst="20121129"
for DAY in $trainDayLst
do
	echo "working on day $DAY"
    type="explore";
    
    mapper="$GPYTHON test.py -m";
    reducer="$GPYTHON test.py -r";
    shipping_file="-file ../vwFeatureExtract.py \
                   -file ../utility.py \
	     		   -file ../test.py \
	               -file ../vowpal_wabbit.py";
    nreducers=1;
    input="/projects/newsrtuf/${type}data_12_auto/ALL/$DAY/finalpvs.out";
    output="/user/wangxl/test_$DAY.out";
    hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar \
                -Dmapred.child.env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./epd/lib" \
                -Dmapreduce.job.cache.archives=$CACHE_PYTHON \
                -Dmapreduce.map.failures.maxpercent=10 \
                -Ddfs.umask=18 \
                -Dmapreduce.output.fileoutputformat.compress=true \
                -Dmapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec \
                -Dmapreduce.map.memory.mb=3072 \
                -Dmapreduce.job.reduces=$nreducers \
                -Dmapreduce.job.queuename=search_fast_lane \
                -mapper $mapper \                
                -reducer $reducer \
                $shipping_file \
                -input $nn$input \
                -output $nn$output
done