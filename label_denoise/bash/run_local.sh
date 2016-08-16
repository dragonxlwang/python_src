#!/bin/sh
source ${HOME}/.profile_wangxl

tmp_folder="label_denoise"
#task_job="run_remote_debug.sh";
#task_job="run_remote_get_mt_annotation.sh";
#task_job="run_remote_ver_dist.sh";
#task_job="run_remote_get_editor_annotation.sh";
#task_job="run_remote_ver_dist_debug.sh";
#task_job="run_remote_avgfv_pig.sh";
#task_job="run_remote_nquery_pig.sh";
task_job="run_remote_debug_pig.sh";
nitro_node="gwbl6107";
nitro_node="nitro-gw";
nitro_blue="wangxl@${nitro_node}.blue.ygrid.yahoo.com";
nitro_blue_ssh="ssh ${nitro_blue}";

src_path="${HOME}/workspace/src/label_denoise/";
src_files=`ls ${src_path} | grep -E ".py$"`;
pig_files=$(echo $(ls ${src_path}pig | grep -E ".pig$") | \
			gsed -r "s@\S+@pig/&@g");

redecho "clean/build tmp directory...";
$nitro_blue_ssh "rm -rfv $tmp_folder; mkdir -vp $tmp_folder"
cur_pwd=${PWD}
cd $src_path
# wangxl@nitro-gw.blue.ygrid.yahoo.com:label_denoise/

redecho "copying source files...";
scp $(printf "$src_files\n$pig_files") "${nitro_blue}:${tmp_folder}/"
cd $cur_pwd

# run
redecho "executing task on Nitro Blue...";
$nitro_blue_ssh < $task_job
$nitro_blue_ssh "rm -rfv $tmp_folder"
