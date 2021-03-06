#!/bin/sh
#source
source ~/.bash_profile
cd ~/src
# ticket
kinit <<< PASSWORD@

# svn update
#svn update

# data params
src_path="${HOME}/label_denoise/";
trainDayLst="20121129 20121130 20121201 20121202 20121203 20121204 20121204 20121205 20121206 20121207 20121208 20121209 20121210 20121211 20121213";
#trainDayLst="20121129"
type="explore"; 

# hadoop system params
CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON="epd/bin/python"
nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com
hadoop=/home/gs/hadoop/current/bin/hadoop;

#hadoop params   
mapper="$GPYTHON test.py --get_mt_annotation -m";
reducer="$GPYTHON test.py --get_mt_annotation -r";
	# reducer="cat";
shipping_file_param_str=$(echo $(ls ${src_path}) | sed -r "s@\S+@-file ${src_path}&@g");
	# input="/projects/newsrtuf/${type}data_12_auto/ALL/$DAY/finalpvs.out";
input_files_param_str=$(echo $trainDayLst | \
					   sed -r "s@\S+@-input ${nn}/user/wangxl/data/data_with_tag/&/finalpvs_annotation.out@g");
output_file="${nn}/user/wangxl/mt.out";
output_file_param_str="-output ${output_file}";
#check
echo "shipping files:";
echo ${shipping_file_param_str};
echo "input files:";
echo ${input_files_param_str};
echo "output file:"
echo ${output_file_param_str};
# clean
hadoop fs -rm -r ${output_file}
nreducers=1;
    
#run    
    $hadoop jar $HADOOP_PREFIX/share/hadoop/tools/lib/hadoop-streaming.jar \
                -Dmapred.child.env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./epd/lib" \
                -Dmapreduce.job.cache.archives=$CACHE_PYTHON \
                -Dmapreduce.map.failures.maxpercent=10 \
                -Ddfs.umask=18 \
                -Dmapreduce.output.fileoutputformat.compress=true \
                -Dmapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec \
                -Dmapreduce.map.memory.mb=3072 \
                -Dmapreduce.job.reduces=$nreducers \
                -Dmapreduce.job.queuename=search_fast_lane \
                -mapper "$mapper" \
                -reducer "$reducer" \
                $shipping_file_param_str \
            	$input_files_param_str \
            	$output_file_param_str 
                
