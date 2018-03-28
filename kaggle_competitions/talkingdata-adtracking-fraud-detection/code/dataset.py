#!/usr/bin/env python

# author: set_daemon@126.com
# date: 2018-03-12

import numpy as np
import pandas as pd

class Dataset(object):
	def __init__(self, train_ds_file="../inputs/train.csv", test_ds_file="../inputs/test.csv"):
		self.train = {
			"file": train_ds_file,
			"headers": ["ip","app","device","os","channel","click_time","attributed_time","is_attributed"],
			"dtypes": {
				"ip": np.uint32,
				"app": np.uint16,
				"device": np.uint16,
				"os": np.uint16,
				"channel": np.uint16,
				"is_attributed": np.int8,
			},
		}

		self.test = {
			"file": test_ds_file,
			"headers": ["click_id", "ip", "app", "device", "os", "channel", "click_time"],
			"dtypes": {
				"ip": np.uint32,
				"app": np.uint16,
				"device": np.uint16,
				"os": np.uint16,
				"channel": np.uint16,
			},
		}

	def iterate_train(self, nrows=10000, use_columns=None):
		skip_rows = 1
		while True:
			df = pd.read_csv(self.train["file"], sep=',', header=None, names=self.train["headers"], dtype=self.train["dtypes"], skiprows=skip_rows, nrows=nrows, usecols=use_columns)
			if df is None or len(df) <= 0:
				break
			data_rows = len(df)
			skip_rows += data_rows
			
			yield df

	def iterate_test(self, nrows=10000, use_columns=None):
		skip_rows = 1
		while True:
			df = pd.read_csv(self.test["file"], sep=',', header=None, names=self.test["headers"], dtype=self.test["dtypes"], skiprows=skip_rows, nrows=nrows, usecols=use_columns)
			if df is None or len(df) <= 0:
				break
			data_rows = len(df)
			skip_rows += data_rows
			yield df


	def get_partial_test(self, skip_rows=1, max_rows = 100000, data_fields=None):
		df = pd.read_csv(self.test["file"], sep=',', header=None, names=self.test["headers"], dtype=self.test["dtypes"], skiprows=skip_rows, nrows=max_rows, usecols=data_fields)

		return df

	def get_partial_train(self, skip_rows=1, max_rows = 100000, data_fields=None):
		df = pd.read_csv(self.train["file"], sep=',', header=None, names=self.train["headers"], dtype=self.test["dtypes"], skiprows=skip_rows, nrows=max_rows, usecols=data_fields)

		return df
