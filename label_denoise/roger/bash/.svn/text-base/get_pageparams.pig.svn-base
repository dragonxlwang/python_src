-- get_features.pig
-- extracts a bunch of common features from the ULT records

register /grid/0/gs/pig/latest/lib/sds.jar;
-- setup some default values if we don't have values from a conf file.
%default input_view_source '/data/SDS/data/search_US/20100828/18025/view/H629/part-00078'
%default input_robot_source '/data/SDS/data/robotlist_yahoo/20100828/'
%default default_parallelism 15
%default output_dir 'test/ult'
%default qlas_config_proxy 'config.proxy.us.xml'
%default qlas_data_pack '/projects/qlas_on_grid/qlas_data_us-20100813.jar#datapack' 
%default qlas_libs '/projects/qlas_on_grid/qlas_libs-1.2.3.jar#qlas'

-- setup some common defines that we will be using regularly.
define ULTLoader com.yahoo.yst.sds.ULT.ULTLoader();

--define qlas_analyze `mapper.sh` ship('mapper.sh') ship('$qlas_config_proxy') cache('$qlas_data_pack') cache('$qlas_libs');

-- load the various inputs, project necessary fields and do BOT filtering.
input_views = load '$input_view_source' using ULTLoader as (sf:[], mf:[], mlf:[]);
input_robots = load '$input_robot_source' using ULTLoader as (sf:[], mf:[], mlf:[]);

filtered_input_v = filter input_views by
	(sf#'bcookie' is not null) and
	(sf#'bcookie' != '')	and
	(sf#'timestamp' != '')	and
	(sf#'srcpvid' is not null) and
	(sf#'srcpvid' !='') and
	(mf#'page_params'#'query' != '') and
	(mlf#'viewinfo' is not null) and
	(mf#'page_params'#'pagenum' == 1) and
	(sf#'ydod' is null) and
	(mf#'page_params'#'sltmod' == 'northstar');

view_data = foreach filtered_input_v generate
	(chararray) sf#'bcookie' as bcookie,
	(chararray) sf#'srcpvid' as srcpvid,
	(chararray) mf#'page_params'#'query' as query,
	(chararray) mf#'page_params'#'qpddinfo' as qpddinfo,
	(chararray) mf#'page_params'#'ddinfo' as ddinfo,
	(chararray) mf#'page_params'#'yfed' as yfed,
    (chararray) mf#'page_params'#'nsinfo' as nsinfo,
    (chararray) mf#'page_params'#'nsddo' as nsddo;

queries = foreach view_data generate query;
queries = distinct queries parallel $default_parallelism;

robot_data = foreach input_robots generate (chararray)sf#'bcookie' as bcookie;
bot_filtering_1 = cogroup view_data by bcookie, robot_data by bcookie parallel $default_parallelism;
bot_filtering_2 = filter bot_filtering_1 by IsEmpty(robot_data.bcookie);
fult_raw_feats = foreach bot_filtering_2 generate flatten(view_data);
fult_raw_feats = foreach fult_raw_feats generate srcpvid, query, '""' as qlas, qpddinfo, ddinfo, yfed, nsinfo, nsddo;

-- store the output data somewhere - curently at output /ult_features
-- queries = foreach fult_raw_feats generate query;
-- interpretations = stream queries through qlas_analyze;
--store queries into '$output_dir/queries' ;
--store fult_raw_feats into '$output_dir/ult_features';

-- now load the rewards features generated inside $output_dir/session_rewards and join them with the fult_raw_feats
rewards = load '$output_dir/session_rewards' using PigStorage('\t') as (srcpvid:chararray, 
	seed:chararray, 
	bcookie:chararray, 
	timestamp:long, 
	ig1:chararray, ig2:chararray, ig3:chararray, 
	action:chararray,
	reward:float,
	weight:float);

grewards = group rewards by srcpvid parallel $default_parallelism;
grewards = foreach grewards generate flatten(rewards);

rewards_and_fult = join grewards by srcpvid, fult_raw_feats by srcpvid parallel $default_parallelism;
raw_pvobjs = foreach rewards_and_fult generate grewards::rewards::srcpvid as id, seed, grewards::rewards::bcookie as bc, timestamp, action, reward, weight, query, qlas, qpddinfo, ddinfo, yfed, nsinfo, nsddo;

store raw_pvobjs  into '$output_dir/raw_pvobjs' using PigStorage('\u0001');
