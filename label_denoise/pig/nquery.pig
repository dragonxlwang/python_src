-- nquery.pig
-- average fv per nquery

-- setup some default values if we don't have values from a conf file.
%DEFAULT cache_python '/user/luojie/cache/epd.tgz#epd'
%DEFAULT GPYTHON 'epd/bin/python'
%DEFAULT input_files_list '/user/wangxl/data/data_with_tag/20121129/finalpvs.out'
%DEFAULT output_file '/user/wangxl/avg_fv.out'
%DEFAULT shipping_file_list '/user/wangxl/label_denoise/avgfv.pig'

-- setup some common defines that we will be using regularly.
DEFINE get_nquery `$GPYTHON label_denoise/test.py --get_nquery` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));
DEFINE avgfv `$GPYTHON label_denoise/test.py --avg_fv` INPUT(stdin USING PigStreaming('\u0001')) OUTPUT(stdout USING PigStreaming('\u0001'));

data = LOAD '$input_files_list' USING PigStorage('\u0001') AS (id, rec);
data = FILTER data BY (SIZE(id) > 0) AND (SIZE(rec) > 0);
extracted_query_rec = STREAM data THROUGH get_nquery AS (nquery, id, rec);
query_rel = FOREACH extracted_query_rec GENERATE nquery;
query_group = GROUP query_rel BY nquery;
query_cnt = FOREACH query_group GENERATE group AS nquery, COUNT(query_rel) AS cnt;
query_cnt = ORDER query_cnt BY cnt DESC;
STORE query_cnt INTO '$output_file' USING PigStorage('\u0001');