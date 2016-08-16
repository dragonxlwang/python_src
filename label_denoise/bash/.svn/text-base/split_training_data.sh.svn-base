#! /bin/sh
trainDayLst="20121129 20121130 20121201 20121202 20121203 20121204 20121204 20121205 20121206 20121207 20121208 20121209 20121210 20121211 20121213";
type="explore";


# hadoop params
CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON="epd/bin/python"
nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com
nreducers=3;


for DAY in $trainDayLst
do
	echo "Process Day $DAY"
	ship_files="-file data_process.py";
	# run
	if [ $1 = "split" ]
	then
		echo "spliting...";
		mapper="$GPYTHON data_process.py -a";
		input="/projects/newsrtuf/${type}data_12_auto/ALL/$DAY/finalpvs.out";
		output="/user/wangxl/data/data_with_tag/$DAY/finalpvs.out";
		# clean
		hadoop fs -rm -r $nn$output
	elif [ $1 = "click" ]
	then
		echo "click data..."
		mapper="$GPYTHON data_process.py -o click";
		input="/user/wangxl/data/data_with_tag/$DAY/finalpvs.out";
		output="/user/wangxl/data/data_with_tag/$DAY/finalpvs_click.out";
		# clean
		hadoop fs -rm -r $nn$output
	elif [ $1 = "annotation" ]
	then
		echo "annotation data..."
		mapper="$GPYTHON data_process.py -o annotation";
		input="/user/wangxl/data/data_with_tag/$DAY/finalpvs.out";
		output="/user/wangxl/data/data_with_tag/$DAY/finalpvs_annotation.out";
		# clean
		hadoop fs -rm -r $nn$output
	fi


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
				-mapper "$mapper" \
				-reducer cat \
				$ship_files \
				-input $nn$input \
				-output $nn$output
done
