#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.externals import joblib

import psutil
import os

def memory_usage(info):
	mmount = psutil.Process(os.getpid()).memory_info().rss
	print u'[%s] 内存使用 %fByte,%fKB,%fMB' %(info, mmount, mmount/1024.0, mmount/1024.0/1024)

dtypes = {
	"ip": np.uint32,
	"app": np.uint16,
	"device": np.uint16,
	"os": np.uint16,
	"channel": np.uint16,
	"is_attributed": np.int8,
}

use_columns = ["ip", "app", "device", "os", "channel", "is_attributed"]
train_headers = ["ip","app","device","os","channel","click_time","attributed_time","is_attributed"]
skip_rows = 1
# 320万OK 8G内存 --- tested
nrows = 320*10000
memory_usage('before load train.csv')
df = pd.read_csv('../inputs/train.csv', sep=',', header=None, names=train_headers, dtype=dtypes, usecols=use_columns, nrows=nrows, skiprows=skip_rows)
memory_usage('after load train.csv')

memory_usage('before reshape ips')
ips = df["ip"].unique()
ips = ips.reshape(ips.shape[0], 1)
memory_usage('after reshape ips')

memory_usage('before reshape apps')
apps = df["app"].unique()
apps = apps.reshape(apps.shape[0], 1)
memory_usage('after reshape apps')

memory_usage('before reshape devices')
devices = df["device"].unique()
devices = devices.reshape(devices.shape[0], 1)
memory_usage('after reshape devices')

memory_usage('before reshape oses')
oses = df["os"].unique()
oses = oses.reshape(oses.shape[0], 1)
memory_usage('after reshape oses')

memory_usage('before reshape channels')
channels = df["channel"].unique()
channels = channels.reshape(channels.shape[0], 1)
memory_usage('after reshape channels')

memory_usage('before OneHotEncoder')
ip_encoder = OneHotEncoder()
ip_encoder.fit(ips)

app_encoder = OneHotEncoder(sparse=False, dtype=np.int8)
app_encoder.fit(apps)

device_encoder = OneHotEncoder()
device_encoder.fit(devices)

os_encoder = OneHotEncoder()
os_encoder.fit(oses)

channel_encoder = OneHotEncoder()
channel_encoder.fit(channels)
memory_usage('after OneHotEncoder')

# model training
memory_usage('before dataset division')
df_rows = df.shape[0]
## simple division of train & test
test_size = int(df_rows * 3 / 5)
train_df = df[0:test_size]
train_test_df = df[test_size:]
memory_usage('after dataset division')

# app
memory_usage('before transform dataframe to numpy.array')
train_x = np.array(train_df[["app"]], dtype=np.uint16)
memory_usage('after transform dataframe to numpy.array')
memory_usage('before OneHotEncoder encoding data')
train_x = app_encoder.transform(train_x)
memory_usage('after OneHotEncoder encoding data')
memory_usage('before transform dataframe to numpy.array')
train_y = np.array(train_df["is_attributed"], dtype=np.int8)
memory_usage('after transform dataframe to numpy.array')

memory_usage('before transform dataframe to numpy.array')
train_test_x = np.array(train_test_df[["app"]], dtype=np.uint16)
memory_usage('after transform dataframe to numpy.array')
memory_usage('before OneHotEncoder encoding data')
train_test_x = app_encoder.transform(train_test_x)
memory_usage('after OneHotEncoder encoding data')
memory_usage('before transform dataframe to numpy.array')
train_test_y = np.array(train_test_df["is_attributed"], dtype=np.int8)
memory_usage('after transform dataframe to numpy.array')

memory_usage('before create LR')
lr = LogisticRegression()
memory_usage('after create LR')
memory_usage('before LR train')
lr.fit(train_x, train_y)
memory_usage('after LR train')

memory_usage('before LR predict')
train_y_pred = lr.predict(train_x)
memory_usage('after LR predict')
memory_usage('before LR predict')
train_test_y_pred = lr.predict(train_test_x)
memory_usage('after LR predict')

train_y_pred_logloss = log_loss(train_y, train_y_pred)
print set(train_test_y.tolist())
train_test_y_pred_logloss = log_loss(train_test_y, train_test_y_pred)
print "train logloss = %f, train test logloss = %f" %(train_y_pred_logloss, train_test_y_pred_logloss)

# store the model
joblib.dump(lr, '../outputs/lr_320w_app.pkl')

memory_usage('before clear data')
# try to clear the trained data to see if it can shrink the memory ?
del df
memory_usage('after clear data')

# 计算测试集
dtypes = {
	"ip": np.uint32,
	"app": np.uint16,
	"device": np.uint16,
	"os": np.uint16,
	"channel": np.uint16,
}

use_columns = ["click_id", "ip", "app", "device", "os", "channel"]
test_headers = ["click_id", "ip", "app", "device", "os", "channel"]
skip_rows = 1
nrows = 100*10000

submission_file = "../outputs/submission.csv"
sub_f = file(submission_file, 'w')
sub_f.write('click_id,is_attributed\n')

while True:
	test_df = pd.read_csv('../inputs/test.csv', sep=',', header=None, names=test_headers, dtype=dtypes, usecols=use_columns, nrows=nrows, skiprows=skip_rows)
	if test_df is None or len(test_df) <= 0:
		break	
	data_rows = len(test_df)
	
	memory_usage('before app encoder %d rows' %(data_rows))
	test_apps = np.array(test_df[["app"]], dtype=np.uint16)
	test_apps = app_encoder.transform(test_apps)
	memory_usage('after app encoder %d rows' %(data_rows))
	
	memory_usage('before predict test rows = %d' %(data_rows))
	test_pred = lr.predict_proba(test_apps)
	memory_usage('after predict test rows = %d' %(data_rows))

	memory_usage('before save prediction datas rows = %d' %(data_rows))
	result_cache = ''
	for i in range(0, data_rows):
		result_cache += '%s,%s\n' %(test_df["click_id"][i], test_pred[i][1])
	
		if i % 2000 == 0:
			sub_f.write(result_cache)
			sub_f.flush()
			result_cache = ''
	if result_cache != '':
		sub_f.write(result_cache)

	del test_df
	memory_usage('after save prediction datas rows = %d' %(data_rows))

	skip_rows += data_rows

sub_f.close()

if __name__ == "__main__":
	pass
