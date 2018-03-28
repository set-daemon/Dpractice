#!/bin/bash
# set_daemon@126.com
# 2018-03-15

log_file="../outputs/train.log"
start_time=`date "+%Y%m%d %H:%M:%S"`
# 20180314 completed
#feature="app"
# 20180315 executing
#feature="os"
#feature="device"
feature="channel"

echo "[${start_time}]======================start to train ${feature}======================" >> ${log_file}
python lr_train.py ${feature}
end_time=`date "+%Y%m%d %H:%M:%S"`
echo "[${end_time}]======================end to train ${feature}======================" >> ${log_file}
