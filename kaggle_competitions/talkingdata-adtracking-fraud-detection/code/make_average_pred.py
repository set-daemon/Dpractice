#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-12

import sys
import os
import time
import re

import numpy as np
import pandas as pd

# the destination is to calculate the average value of same column from multiple files
# example: files A,B,C has the same columns 'click_id','is_attributed', 
#  then we want get the average value of (A(is_attributed)+B(is_attributed)+C(is_attributed))/3
#  but problem comes as we have no enough memory and cpu resources for calculating, so how
#  to solve it?

def calculate_average(args, out_file):
	file_num = 0
	result = None
	for csv_file in args:
		df = pd.read_csv(csv_file, header=0, sep=',')
		if df is None or len(df) <= 0:
			continue
		file_num += 1
		print "calculated %s" %(csv_file)
		if result is None:
			result = df
			continue
		result["is_attributed"] += df["is_attributed"]

		del df

	result["is_attributed"] /= file_num

	result.to_csv(out_file, sep=',', index=False, columns=['click_id', 'is_attributed'])

	return True

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "not enough arguments: out_file data_dir"
		sys.exit(0)
	data_dir = sys.argv[1]
	out_file = sys.argv[2]
	feature = sys.argv[3]

	f_pattern = re.compile(r'lr_model_%s_\d+_test_pred\.csv' %(feature))

	data_files = []
	file_list = os.listdir(data_dir)
	for i in range(0, len(file_list)):
		fname = file_list[i]
		if len(f_pattern.findall(fname)) <= 0:
			continue
		data_files.append('%s/%s' %(data_dir, fname))

	calculate_average(data_files, out_file)

	print "average calculation completed"
