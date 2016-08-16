#! /bin/sh
# ./vwexamples.mapper.sh [local|remote] inputdir outprefix
ship_files="-file vwexamples.mapper.sh -file afsdist.tar.gz -file $4 -file $5"
script="vwExampleGenerator.py --keep-features $5 -v w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,vn,vm,vl,vs --fstats $4"
if [ $# -eq 6 ]
then
    script="vwExampleGenerator.py -t -0 --keep-features $5 -s s1.1 -v w1,vn,vm,vl,vs --fstats $4"
    nreducers=10
else
    nreducers=3
fi

CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON="epd/bin/python"

nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com
input=$2
outprefix=$3 
featmeta=$4
if [ $1 == "local" ]
then
    echo "Are you shipping afsdist.tar.gz with latest changes?"
    id=$RANDOM
    echo "Running job for INPUT: $input"
    hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar -Dmapred.child.env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./epd/lib" -Dmapreduce.job.cache.archives=$CACHE_PYTHON -Dmapreduce.map.failures.maxpercent=10 -Ddfs.umask=18 -Dmapreduce.output.fileoutputformat.compress=true -Dmapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec -Dmapreduce.map.memory.mb=3072 -Dmapreduce.job.reduces=$nreducers -Dmapreduce.job.queuename=search_fast_lane -cacheFile $nn/projects/newsrtuf/cache/python-2.6.4.tar.gz#pydist -mapper "vwexamples.mapper.sh remote $2 $3 $4 $5 $6" -reducer cat $ship_files  -input $nn$input -output $nn$outprefix.out

elif [ $1 == "remote" ]
then    
    tar -zxf pydist
    tar -zxf afsdist.tar.gz
    py26=$GPYTHON
    echo -e "python executable: `ls -l $py26`" >&2
    (($py26 $script) 3>&1 1>&2 2>&3  | tee --append log.err.txt) 3>&1 1>&2 2>&3
    fname="log.`hostname`_$RANDOM.err.txt"
    hadoop fs -mkdir $nn$outprefix.logs
    hadoop fs -put log.err.txt $nn$outprefix.logs/$fname
fi

