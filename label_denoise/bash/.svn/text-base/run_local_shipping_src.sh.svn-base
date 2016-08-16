#!/bin/sh
source ${HOME}/.profile_wangxl

tmp_folder="label_denoise"
nitro_node="gwbl6107";
nitro_node="nitro-gw";
nitro_blue="wangxl@${nitro_node}.blue.ygrid.yahoo.com";
nitro_blue_ssh="ssh ${nitro_blue}";
timan="xwang95@timan101.cs.illinois.edu";
timan_ssh="ssh ${timan}";
src_path="${HOME}/workspace/src/label_denoise/";
src_files=`ls ${src_path} | grep -E ".py$"`;
pig_files=$(echo $(ls ${src_path}pig | grep -E ".pig$") | \
			gsed -r "s@\S+@pig/&@g");
bash_files=$(echo $(ls ${src_path}bash | grep -E ".sh$") | \
			gsed -r "s@\S+@bash/&@g");

redecho "clean/build tmp directory...";
yellowecho "nitro_blue"
$nitro_blue_ssh "rm -rfv $tmp_folder; mkdir -vp $tmp_folder"
yellowecho "timan"
$timan_ssh "rm -rfv $tmp_folder; mkdir -vp $tmp_folder"
cur_pwd=${PWD}
cd $src_path
# wangxl@nitro-gw.blue.ygrid.yahoo.com:label_denoise/

redecho "copying source files...";
yellowecho "nitro_blue"
scp $(printf "$src_files\n$pig_files\n$bash_files") "${nitro_blue}:${tmp_folder}/"
yellowecho "timan"
scp $(printf "$src_files\n$pig_files\n$bash_files") "${timan}:${tmp_folder}/"
cd $cur_pwd
