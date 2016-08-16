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
DEFINE pig_stream_gather_click_data_after_group `$GPYTHON label_denoise/test.py --pig_stream_gather_click_data_after_group` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));

nquery_layout_weight_rel = LOAD 'exp/cache_model_1edi_1mt_user_9_iter_post_estimate_pig' USING PigStorage('\u0001') AS (nquery, layout, weight);
nquery_avgfv_rel = LOAD 'exp/avg_fv.out' USING PigStorage('\u0001') AS (nquery, avgfv);
nquery_layout_weight_nquery_avgfv = JOIN nquery_layout_weight_rel BY nquery, nquery_avgfv_rel BY nquery;
nquery_layout_weight_avgfv_rel = FOREACH nquery_layout_weight_nquery_avgfv GENERATE nquery_layout_weight_rel::nquery, nquery_layout_weight_rel::layout, nquery_layout_weight_rel::weight, nquery_avgfv_rel::avgfv;
STORE nquery_layout_weight_avgfv_rel INTO 'exp/nquery_layout_weight_avgfv.out.gz' USING PigStorage('\u0001');