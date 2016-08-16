#!/bin/bash
type="explore"
bucketId=ALL
py26=/homes/luojie/epd/bin/python
trainDIR_Base="/projects/newsrtuf/${type}data_13_nf/${bucketId}/{*}/vwexamples.allslots.adjw_2.newlocal.bigram/train.out"
#trainDIR_Base="/projects/newsrtuf/${type}data_13_nf/${bucketId}/201212{*}/vwexamples.allslots.adjw_2.newlocal.bigram/train.out"
testDIR_Base="/projects/newsrtuf/${type}data_13_nf/${bucketId}/{*}/vwexamples.allslots.adjw_2.newlocal.bigram/test.out"
#dayList=`hadoop fs -ls $afsDataDIR_New | cut -f 6 -d '/' | grep -v dist | grep 201[2-9] | tail -n 1  | tr '\n' ' '`
theday="newlocalshopping.bigram"

# Model training parameter
onlinepasses=1
batchpasses=50
regularization="100"
nbits="23"
quadcomb="aq,al"
modeltag="ns13"
model_directory=/user/luojie/model

function run {
	day=$1
	sh train_test.sh $trainDIR_Base $testDIR_Base $onlinepasses $batchpasses $regularization $nbits $modeltag ${model_directory}/${day} \"$quadcomb\" 

	hadoop fs -chmod -R 777 $model_directory/$day
}

cd /homes/luojie/northstar/ns-auto-pl/afs-dist
echo "Train $theday model" 
run $theday 


