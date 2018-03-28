#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-12

import sys
import os
import time

import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from dataset import Dataset as Xdataset
from features import DatasetFeatures

from worker import Worker 



def predWorkerCb(msg, ctx):
	feature_mgr = ctx["featureMgr"]
	model = ctx["model"]
	outpath = ctx["outpath"]
	x_ds = ctx["ds"]
	feature = ctx["feature"]

	skip_rows = msg["skip_rows"]
	max_rows = msg["max_rows"]

	data_fields = ['click_id', feature]
	df = x_ds.get_partial_test(skip_rows=skip_rows, max_rows=max_rows, data_fields=data_fields)
	if df is None or len(df) <= 0:
		return {'type': "fail", 'reason':'no more data'}

	data_rows = len(df)

	test_datas = np.array(df[[feature]], dtype=np.uint16)
	x_fea_encoders = feature_mgr.getFeatureOneHotEncoder([feature])
	test_encoded_data = x_fea_encoders[feature].transform(test_datas)

	predicted_data = model.predict_proba(test_encoded_data)

	#print "worker %d, task id %d" %(os.getpid(), msg["task_id"])
	# save the predicted result to file
	out_file = "%s/worker_%d_task_%d" %(outpath, os.getpid(), msg["task_id"])
	out_f = file(out_file, 'w')

	tmp_lines = ""
	line_num = 0
	for i in range(0, data_rows):
		tmp_lines += '%d,%.15f' %(df["click_id"][i], predicted_data[i][1])
		if i < data_rows - 1:
			tmp_lines += '\n'
		line_num += 1

		if line_num == 10000:
			out_f.write(tmp_lines)
			out_f.flush()
			tmp_lines = ""
			line_num = 0

	if tmp_lines != "":
		out_f.write(tmp_lines)
		out_f.flush()

	out_f.close()

	del df
	del test_encoded_data
	return {"type": "ok", "task_id": msg["task_id"], "file": out_file, 'data_rows':data_rows}

def dispatcher(args):
	'''
	argument list: feature trained_model_file out_file
	'''
	if len(args) < 3:
		print "not enough arguments"
		sys.exit(0)

	feature = args[0]
	model_file = args[1]
	pred_out_file = args[2]

	workers = []
	out_path = '../outputs/pred'
	if os.path.exists(out_path) is not True:
		os.makedirs(out_path)

	model = joblib.load(model_file)
	x_ds = Xdataset()
	x_ds_features = DatasetFeatures(feature_path='../basic_features')

	start_time = time.time()

	ctx = {'feature': feature, 'ds':x_ds, "model": model, "outpath": out_path, "featureMgr":x_ds_features}
	# get cpu number for creating workers
	worker_num = Worker.suitable_worker_num()
	# create workers
	for i in range(0, worker_num):
		worker = Worker("predWorker", predWorkerCb, ctx)
		worker.start()
		workers.append(worker)

	total_rows = 0
	# dispatch tasks, monitor workers
	max_rows = 100000
	skip_rows = 1
	tasks = []
	task_id = 0
	max_task_num = -1
	no_tasks = False
	while True:
		# check any free worker
		for worker in workers:
			if no_tasks is not True and worker.is_free() and (max_task_num < 0 or task_id < max_task_num):
				# create task & dispatch it to this worker	
				msg = {
					"type": "task",
					'skip_rows' : skip_rows,
					'max_rows' : max_rows,
					'task_id' : task_id,
				}
				task_id += 1
				worker.send(msg)
				tasks.append("")
				skip_rows += max_rows

		if max_task_num > 0 and task_id >= max_task_num:
			no_tasks = True

		free_worker_num = 0
		for worker in workers:
			if worker.is_busy():
				msg = worker.recv()
				if msg is False:
					continue
				if msg["type"] == "ok":
					pred_file = msg["file"]
					rsp_task_id = msg["task_id"]
					tasks[rsp_task_id] = pred_file
					total_rows += msg["data_rows"]
				elif msg["reason"] == "no more data":
					no_tasks = True		
				else:
					pass

				free_worker_num += 1	
			else:
				free_worker_num += 1	

		if no_tasks is True and free_worker_num == worker_num:
			break

	# offline these workers
	for worker in workers:
		worker.stop()

	# merge the result
	out_f = file(pred_out_file, "w")
	out_f.write('click_id,is_attributed\n')
	task_num = len(tasks)
	print "task num = %d, task_id = %d, total data rows = %d" %(task_num, task_id, total_rows)
	for i in range(0, task_num):
		result_file = tasks[i]
		if result_file == '': # the task that read no datas will result an empty output file
			print 'task id %d with empty file' %(i)
			continue
		in_f = file(result_file, 'r')
		content = in_f.read()
		in_f.close()
		os.remove(result_file)

		if i < task_num - 1: 
			out_f.write("%s\n" %(content))
		else:
			out_f.write("%s" %(content))

		out_f.flush()

	out_f.close()
	print "completed %f seconds, model %s" %(time.time() - start_time, model_file)

if __name__ == "__main__":
	dispatcher(sys.argv[1:])
