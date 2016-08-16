#!/bin/sh

fea_file_path=$1
fold_num=$2
if [[ $# -eq 3 ]]
then
	select_fold=$3;
else
	select_fold=-1;
fi

if [[ $select_fold -eq -1 ]]
then
	for (( i=1;i<=$fold_num;i++ ))
	do
		r=$((i-1));
		train_file_fold_path="${fea_file_path}_${i}_train";
		test_file_fold_path="${fea_file_path}_${i}_test";
		echo "generating fold ${i}";
		echo "train file path: ${train_file_fold_path}";
		echo "test file path: ${test_file_fold_path}";
		echo;
		awk '{if(NR%fn != r) print $0}' r=$r fn=$fold_num $fea_file_path > $train_file_fold_path;
		awk '{if(NR%fn == r) print $0}' r=$r fn=$fold_num $fea_file_path > $test_file_fold_path;
	done
else
	i=$select_fold;
	r=$((i-1));
	train_file_fold_path="${fea_file_path}_${i}_train";
	test_file_fold_path="${fea_file_path}_${i}_test";
	echo "generating fold ${i}";
	echo "train file path: ${train_file_fold_path}";
	echo "test file path: ${test_file_fold_path}";
	echo;
	awk '{if(NR%fn != r) print $0}' r=$r fn=$fold_num $fea_file_path > $train_file_fold_path;
	awk '{if(NR%fn == r) print $0}' r=$r fn=$fold_num $fea_file_path > $test_file_fold_path;
fi