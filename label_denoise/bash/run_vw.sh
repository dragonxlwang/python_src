#!/bin/sh

online_pass_num=10;
batch_pass_num=10;
nbits=20;
regularization=10;
train_file="${HOME}/exp/nquery_layout_weight_avgfv.out_1_train";
train_file="${HOME}/exp/nquery_layout_weight_avgfv.out_1_test";


vw="${HOME}/misc/vowpal_wabbit/vw";

yellowecho "vw training: sgd ## passes=${online_pass_num}"
$vw --bit_precision $nbits \
	--save_per_pass \
	--adaptive \
	--exact_adaptive_norm \
	--cache_file vw_sgd_cache.tmp \
	--passes $online_pass_num \
	--data $train_file \
	--final_regressor vw_sgd_model \
	--loss_function logistic \
	2>&1 | tee vw_sgd.log
	
yellowecho "vw training: l-bfgs ## passes=${batch_pass_num}"
$vw --bit_precision $nbits \
	--save_per_pass \
	--cache_file vw_bfgs_cache.tmp \
	--passes $batch_pass_num \
	--regularization=$regularization \
	--data $train_file \
	--final_regressor vw_bfgs_model \
	--initial_regressor vw_sgd_model \
	--bfgs \
	--mem 15 \
	--loss_function logistic \
	2>&1 | tee vw_bfgs.log