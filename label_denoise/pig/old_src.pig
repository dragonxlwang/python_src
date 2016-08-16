-- debug.pig
-- debug

-- setup some default values if we don't have values from a conf file.
%DEFAULT cache_python '/user/luojie/cache/epd.tgz#epd'
%DEFAULT GPYTHON 'epd/bin/python'
%DEFAULT input_files_list '/user/wangxl/data/data_with_tag/20121129/finalpvs.out'
%DEFAULT output_file '/user/wangxl/avg_fv.out'
%DEFAULT shipping_file_list '/user/wangxl/label_denoise/avgfv.pig'

-- setup some common defines that we will be using regularly.
DEFINE get_nquery `$GPYTHON label_denoise/test.py --get_nquery` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE avgfv `$GPYTHON label_denoise/test.py --avg_fv` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE extract_nquery_vertical `$GPYTHON label_denoise/test.py --extract_nquery_vertical` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE accum_nquery_vertical `$GPYTHON label_denoise/test.py --accum_nquery_vertical` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));

data = LOAD 'exp/nquery_id_rec_rel.out.gz' USING PigStorage('\u0001') AS (nquery, id, rec);
trans_nquery_rel = FOREACH data GENERATE nquery;
editor_nquery_rel = LOAD 'exp/editor.uniq.nquery.out' AS (nquery);

trans_editor_nquery_rel = JOIN trans_nquery_rel BY nquery, editor_nquery_rel BY nquery;
STORE trans_editor_nquery_rel INTO 'trans_editor_nquery_rel.out.gz' USING PigStorage('\u0001');

group_all = GROUP trans_editor_nquery_rel ALL;
trans_editor_nquery_rel_cnt = FOREACH group_all GENERATE COUNT(trans_editor_nquery_rel);
STORE trans_editor_nquery_rel_cnt INTO 'trans_editor_nquery_rel_cnt.out.gz' USING PigStorage('\u0001');


data = LOAD '$input_files_list' USING PigStorage('\u0001') AS (id, rec);
data = FILTER data BY (SIZE(id) > 0) AND (SIZE(rec) > 0);
nquery_id_rec_rel = STREAM data THROUGH get_nquery AS (nquery, id, rec);
STORE nquery_id_rec_rel INTO 'click_nquery_id_rec_rel.out.gz' USING PigStorage('\u0001');



-- debug.pig
-- debug

-- setup some default values if we don't have values from a conf file.
%DEFAULT cache_python '/user/luojie/cache/epd.tgz#epd'
%DEFAULT GPYTHON 'epd/bin/python'
%DEFAULT input_files_list '/user/wangxl/data/data_with_tag/20121129/finalpvs.out'
%DEFAULT output_file '/user/wangxl/avg_fv.out'
%DEFAULT shipping_file_list '/user/wangxl/label_denoise/avgfv.pig'

-- setup some common defines that we will be using regularly.
DEFINE get_nquery `$GPYTHON label_denoise/test.py --get_nquery` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE avgfv `$GPYTHON label_denoise/test.py --avg_fv` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE extract_nquery_vertical `$GPYTHON label_denoise/test.py --extract_nquery_vertical` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE accum_nquery_vertical `$GPYTHON label_denoise/test.py --accum_nquery_vertical` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE get_trans_with_vertical `$GPYTHON label_denoise/test.py --get_trans_with_vertical` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE pig_stream_collect_click_after_sort `$GPYTHON label_denoise/test.py --pig_stream_collect_click_after_sort` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE pig_stream_get_possible_vertical_per_query `$GPYTHON label_denoise/test.py --pig_stream_get_possible_vertical_per_query` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));



sorted_nquery_id_rec_rel = LOAD 'sorted_nquery_id_rec_rel.out.gz' USING PigStorage('\u0001') AS (nquery, id, rec);
nquery_vset_rel = STREAM sorted_nquery_id_rec_rel THROUGH pig_stream_get_possible_vertical_per_query AS (nquery, vset);
STORE nquery_vset_rel into 'nquery_vset_rel.out.gz' USING PigStorage('\u0001');

click_nquery_id_rec_rel = LOAD 'exp/click_nquery_id_rec_rel.out.gz' USING PigStorage('\u0001') AS (nquery, id, rec);
grouped_data = GROUP click_nquery_id_rec_rel BY nquery;
click_nquery_id_rec_rel = FOREACH grouped_data GENERATE FLATTEN(click_nquery_id_rec_rel);
click_nquery_vidxposrewardlst = STREAM click_nquery_id_rec_rel THROUGH pig_stream_gather_click_data_after_group AS (nquery, vidxposrewardlst);
STORE click_nquery_vidxposrewardlst into 'click_nquery_vidxposrewardlst.out.gz' USING PigStorage('\u0001');