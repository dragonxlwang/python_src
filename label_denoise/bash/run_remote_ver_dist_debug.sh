#!/bin/sh
# ticket
kinit <<< PASSWORD@

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
PYTHON="/homes/luojie/epd/bin/python";
echo "$PYTHON ${src_path}test.py --verDistDebug < ${HOME}/ver_dist.out"
$PYTHON ${src_path}test.py --verDistDebug < ${HOME}/ver_dist.out