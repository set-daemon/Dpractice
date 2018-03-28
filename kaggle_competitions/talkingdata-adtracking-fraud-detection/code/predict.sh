#!/bin/bash

log_file="../outputs/predict.log"
start_time=`date "+%Y%m%d %H:%M:%S"`

feature="channel"
model_path="../outputs/lr_${feature}_1521124526"
for ((i=0; i<=57;i++))
do
	echo "[${start_time}]======================start to predict by feature ${feature} and model ${model_path}model_${i}======================" >> ${log_file}
	python test_lr_pred.py ${feature} ${model_path}/model_${i} ../outputs/pred_results/lr_model_${feature}_${i}_test_pred.csv
	end_time=`date "+%Y%m%d %H:%M:%S"`
	echo "[${end_time}]======================end to predict by feature ${feature} and model ${model_path}model_${i}======================" >> ${log_file}
done

dateH=`date +%Y%m%d%H`
python make_average_pred.py ../outputs/pred_results ../outputs/submission_${feature}_{dateH}.csv ${feature}
