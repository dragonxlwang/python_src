#!/bin/sh
# source
# ticket
source ${HOME}/.profile_wangxl

# data params
src_path="${HOME}/label_denoise/";
trainDayLst="20121129 20121130 20121201 20121202 20121203 20121204 20121204 20121205 20121206 20121207 20121208 20121209 20121210 20121211 20121213";
#trainDayLst="20121129"
type="explore"; 

# hadoop system params
CACHE_PYTHON='/user/luojie/cache/epd.tgz#epd'
GPYTHON='epd/bin/python'
nn=hdfs://nitroblue-nn1.blue.ygrid.yahoo.com
hadoop=/home/gs/hadoop/current/bin/hadoop;

# pig file
#pig_file="${src_path}debug.pig";
pig_file="${src_path}debug.pig";

# shipping source files
blueecho "shipping files:";
src_pkg="label_denoise.tgz";
for file_path in $(echo $(ls ${src_path}));
do
	yellowecho "\t shipping: ${src_path}${file_path}";
done
yellowecho "\t remove old archive (on demand)";
if [ -e "${HOME}/${src_pkg}" ]
then
	rm -rf "${HOME}/${src_pkg}"
fi
yellowecho "\t build archive";
tar -czvf "${HOME}/${src_pkg}" -C ${src_path} $(ls ${src_path});
yellowecho "\t remove archive on HFDS (on demand)";
hadoop fs -test -e ${src_pkg}
if [ $? -eq 0 ]
then
	hadoop fs -rm -r ${src_pkg}
fi
yellowecho "\t ship archive";
hadoop fs -copyFromLocal "${HOME}/${src_pkg}"	
shipping_files_archive='/user/wangxl/label_denoise.tgz#label_denoise';
echo 

# specifying input files
blueecho "input files:";
input_files_list=$(echo $trainDayLst | \
				   sed -r "s@\S+@${nn}/user/wangxl/data/data_with_tag/&/finalpvs_click.out@g");
for input_file in $input_files_list;
do
	yellowecho "\t input: $input_file";
done
input_files_list=$(echo ${input_files_list} | awk '{$1=$1}1' OFS=',');
yellowecho "\t input file parameters:"
yellowecho "\t ${input_files_list}";
echo

#specifying output file
output_file="${nn}/user/wangxl/unused_file";
blueecho "output files:"
yellowecho "\t ${output_file}";
echo

# clean
yellowecho "\t remove output file on HFDS (on demand)";
hadoop fs -rm -r ${output_file}
echo
echo
echo

blueecho "executing pig script ..."        
pig -Dmapred.job.queue.name=search_fast_lane  \
	-Dmapred.map.tasks.speculative.execution=true \
	-Dmapred.reduce.tasks.speculative.execution=true \
	-Dmapred.cache.archives="${CACHE_PYTHON},${shipping_files_archive}" \
	-Dmapred.create.symlink=yes \
	-Dmapred.child.env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./epd/lib:./label_denoise" \
	-param cache_python="${CACHE_PYTHON}" \
	-param GPYTHON="${GPYTHON}" \
	-param input_files_list="${input_files_list}" \
	-param output_file="${output_file}" \
	$pig_file