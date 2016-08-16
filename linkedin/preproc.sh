#!/bin/sh

function func_clean() {
	cd "${HOME}/workspace/src";
	for i in {a..j};
	do
		filename=${HOME}"/data/xa"$i;
		python -u linkedin/tag_proc.py $filename > "${filename}_output" &
	done;	
}

function func_merge() {
	for i in {a..j};
	do
		filename=${HOME}"/data/xa"$i".eng";
		mergename=${HOME}"/data/cleaned_text_eng.txt";
		echo $filename, $mergename;
		cat $filename >> $mergename
	done;
}

function feature_extract() {
	cd "${HOME}/workspace/src";
	for i in {0..9};
	do
		filename=${HOME}"/data/cleaned_text_eng.txt.filtered_by_tag.global0"$i;
		echo $filename;
		python -u linkedin/tag_proc.py $filename 1>"${filename}.fv" &
	done;
}
# clean partitioned files
#func_clean
# merge cleaned files
# func_merge
feature_extract
