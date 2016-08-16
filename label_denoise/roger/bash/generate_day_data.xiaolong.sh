type="explore"
venusDataIn="/projects/newsrtuf/venus/websearch_us/"
afsDataHDFSDirIn=/projects/newsrtuf/${type}data_12_auto
finalPvsPath=/projects/newsrtuf/${type}data_12_auto
afsDataHDFSDirOut=/projects/newsrtuf/${type}data_13
bucketId=ALL
py26=/homes/luojie/epd/bin/python
featuresMetaData=feature-meta-data.txt
qfeatures="vn-rs-feature-whitelist.txt"
whitelist="vn-rs-feature-whitelist.txt"
finalpvs_string="finalpvs"
vw_string="vwexamples.allslots.adjw_2.newlocal.bigram"
genscript="vwexamples.mapper.sh"
trainDayList=`hadoop fs -ls $venusDataIn | cut -f 6 -d '/' | grep -v dist | grep 2012 | tr '\n' ' '`
trainDayList="20121129 20121130 20121201 20121202 20121203 20121204 20121204 20121205 20121206 20121207 20121208 20121209 20121210 20121211 20121213"
testDayList="20121214 20121215"
allDayList=$trainDayList" "$testDayList

mkdir -p logs

function generate {
	day=$1
	test=$2
	echo "Generating data for day "$day
	#sh compute_rewards_venus.sh $bucketId $day $afsDataHDFSDirIn/$bucketId $type
	#sh extract_and_join_fields.sh $bucketId $day $afsDataHDFSDirIn/$bucketId $type
	#sh afs_data_pipe.mapper.sh local $afsDataHDFSDirIn/$bucketId/$day/raw_pvobjs $finalPvsPath/$bucketId/$day/$finalpvs_string  $featuresMetaData $qfeatures 
	if [ $test -eq 1 ] 
	then
		sh $genscript local $finalPvsPath/$bucketId/$day/$finalpvs_string.out $afsDataHDFSDirOut/$bucketId/$day/$vw_string/test $featuresMetaData $whitelist test
        echo
	else
		sh $genscript local \
            $finalPvsPath/$bucketId/$day/$finalpvs_string.out \
            $afsDataHDFSDirOut/$bucketId/$day/$vw_string/train \
            $featuresMetaData $whitelist
        echo
	fi
}


for theday in $allDayList; do 
	testDayMatch=`echo $testDayList | grep -c $theday`
	if [ $testDayMatch -eq 1 ] 
	then
		generate $theday 1 &> logs/data.$vw_string.test.$theday.log &
	else
		generate $theday 0 &> logs/data.$vw_string.train.$theday.log &
	fi
done



