#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-12

import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from dataset import Dataset as Xdataset
from worker import Worker

class DatasetFeatures(object):
	def __init__(self, feature_path="../outputs"):
		self.feature_path = feature_path

	@staticmethod
	def __get_unique_features(msg, ctx):
		skip_rows = msg["skip_rows"]
		max_rows = msg["max_rows"]

		ds = ctx["ds"]
		features = ctx["features"]
		feature_values = {}

		#print "got task %d" %(msg["taskId"])
		df = ds.get_partial_train(skip_rows=skip_rows, max_rows=max_rows, data_fields = features)
		if df is None or len(df) <= 0:
			return {"type":"noMoreData"}

		for feature in features:
			feature_values[feature] = df[feature].unique().tolist()

		del df

		return {"type":"rsp", "taskId": msg["taskId"], "featureValues": feature_values}


	def dump_features_mp(self, feature_list):
		'''
		work under multiple processes
		'''
		worker_num = Worker.suitable_worker_num()
		print "start %d workers" %(worker_num)
		workers = []
		ctx = {
			"ds": Xdataset(),
			"features": feature_list
		}
		for i in range(0, worker_num):
			worker = Worker('feaWorker', DatasetFeatures.__get_unique_features, ctx)
			workers.append(worker)
			worker.start()

		feature_values = {}
		for feature in feature_list:
			feature_values[feature] = []

		print "before assign tasks "
		no_tasks = False
		skip_rows = 1
		max_rows = 100*10000
		task_id = 0
		max_task_num = -1
		while True:
			worker_id = 1
			for worker in workers:
				worker_id += 1
				if no_tasks is not True and worker.is_free() is True and (max_task_num < 0 or task_id < max_task_num):
					msg = {"type":"req", "taskId":task_id, "max_rows": max_rows, "skip_rows":skip_rows}
					worker.send(msg)
					skip_rows += max_rows
					task_id += 1
					print "send task %d to worker %d" %(task_id, worker_id)

			if max_task_num > 0 and task_id >= max_task_num:
				no_tasks = True

			free_worker_num = 0
			for worker in workers:
				if worker.is_busy():
					msg = worker.recv(timeout=1)
					if msg == False:
						continue

					if msg["type"] == "noMoreData":
						no_tasks = True	
					else:
						tmp = msg["featureValues"]
						for feature in tmp.keys():
							feature_values[feature].extend(tmp[feature])
					free_worker_num += 1
				else:
					free_worker_num += 1

			if no_tasks is True and free_worker_num == worker_num:
				print "all task has been done by workers"
				break

		print "task completed"
		for worker in workers:
			worker.stop()

		version = "%d" %(int(time.time()))
		for feature in feature_values.keys():
			fv = feature_values[feature]

			f_filename = '%s/%s_%s.csv' %(self.feature_path, feature, version)
			f = file(f_filename, "w")
			v = sorted(list(set(fv)))
			v = ["%d"%(d) for d in v]
			s = "\n".join(v)
			f.write(s)
			f.close()

		return True

	def dump_features(self, feature_list):
		'''
		work under single process
		'''
		x_ds = Xdataset()	

		feature_values = {}
		for feature in feature_list:
			feature_values[feature] = []

		seq = 0
		# note: if the feature value space is very large, it may cause memory questions.
		for df in x_ds.iterate_train(nrows=100*10000, use_columns=feature_list):
			for feature in feature_list:				
				feature_values[feature].extend(df[feature].unique().tolist())

			if seq % 16 == 0:
				print "%d iterations" %(seq)
			seq += 1
			del df

		print "step 1 completed"

		for feature in feature_list:
			f_filename = '%s/%s.csv' %(self.feature_path, feature)
			f = file(f_filename, "w")
			v = sorted(list(set(feature_values[feature])))
			v = ["%d"%(d) for d in v]
			s = "\n".join(v)
			f.write(s)
			f.close()

		return True

	def load_features(self, feature_list):
		feature_values = {
		}
		for feature in feature_list:
			feature_values[feature] = []	

			f_filename = '%s/%s.csv' %(self.feature_path, feature)
			for line in open(f_filename, 'r'):
				v = line.strip(' \t\n')
				if v == "":
					continue
				v = int(v)
				feature_values[feature].append(v)

		return feature_values

	def getFeatureOneHotEncoder(self, feature_list):
		feature_values = self.load_features(feature_list)

		f_encoders = {}
		for feature in feature_list:
			v = feature_values[feature]
			dtype = np.uint32
			v_len = len(v)
			if v_len <= 0xFF:
				dtype = np.uint8
			elif v_len <= 0xFFFF:
				dtype = np.uint16
			elif v_len <= 0xFFFFFFFF:
				dtype = np.uint32
			else:
				pass
			oh_encoder = OneHotEncoder(dtype=dtype)
			np_arr = np.array(v).reshape(v_len, 1)
			oh_encoder.fit(np_arr)
			f_encoders[feature] = oh_encoder	

		return f_encoders

if __name__ == "__main__":

	ds_fea = DatasetFeatures()
	ds_fea.dump_features_mp(['app', 'device', 'os', 'channel'])
