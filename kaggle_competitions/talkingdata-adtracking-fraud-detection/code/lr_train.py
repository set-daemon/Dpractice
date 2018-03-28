#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-13

import sys
import os
import time
import gc

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from features import DatasetFeatures as DsFeatures
from worker import Worker
from dataset import Dataset as Xdataset

class LrTrainer(object):
	def __init__(self, feature='app', model_outpath="../outputs"):
		version = '%d' %(int(time.time()))
		self.feature = feature
		self.model_outpath = '%s/lr_%s_%s' %(model_outpath, feature, version)
		if os.path.exists(self.model_outpath) is not True:
			os.makedirs(self.model_outpath)

	@staticmethod
	def __train(msg, ctx):
		skip_rows = msg["skip_rows"]
		max_rows = msg["max_rows"]
		model_id = msg["modelId"]

		feature = ctx["feature"]
		fields = ctx["fields"]
		feature_mgr = ctx["featureMgr"]
		ds = ctx["ds"]
		model_outpath = ctx['outpath']

		df = ds.get_partial_train(skip_rows=skip_rows, max_rows=max_rows, data_fields = fields)
		if df is None or len(df) <= 0:
			return {"type":"noMoreData"}

		data_rows = len(df)

		# division of train & test dataset
		train_size = int(data_rows * 4 / 5)
		train_df = df[0:train_size]
		test_df = df[train_size:]

		train_y = np.array(train_df["is_attributed"], dtype=np.int8)
		test_y = np.array(test_df["is_attributed"], dtype=np.int8)

		# now use one feature only	
		feature_encoder = feature_mgr.getFeatureOneHotEncoder([feature])[feature]
		train_x = np.array(train_df[[feature]], dtype=np.uint16)
		train_x_encoded = feature_encoder.transform(train_x)

		model = LogisticRegression()
		model.fit(train_x_encoded, train_y)

		train_y_pred = model.predict(train_x_encoded)

		test_x = np.array(test_df[[feature]], dtype=np.uint16)
		test_x_encoded = feature_encoder.transform(test_x)
		test_y_pred = model.predict(test_x_encoded)

		print 'train logloss = %f' %(log_loss(train_y, train_y_pred))
		print 'test logloss = %f' %(log_loss(test_y, test_y_pred))

		model_file = '%s/model_%d' %(model_outpath, model_id)
		joblib.dump(model, model_file)

		del df
		del train_x
		del test_x
		del test_x_encoded
		del train_x_encoded
		del train_y_pred
		del test_y_pred
		gc.collect()

		return {"type":"ok", 'modelFile':model_file}

	def train_mp(self):
		worker_num = Worker.suitable_worker_num()
		workers = []

		ds = Xdataset()
		feature_mgr = DsFeatures(feature_path='../basic_features')

		ctx = {'feature':self.feature, 'fields': [self.feature, 'is_attributed'], 'trainDataMinSize':300*10000, 'outpath': self.model_outpath, 'featureMgr':feature_mgr, 'ds':ds}

		for i in range(0, worker_num):
			worker = Worker('lrTrainWorker', LrTrainer.__train, ctx)
			workers.append(worker)
			worker.start()

		start_time = time.time()

		skip_rows = 1
		max_rows = 320*10000
		#max_rows = 10*10000
		model_id = 0
		max_model_num = -1
		no_tasks = False
		models = []
		print "start to assign tasks"
		while True:
			for worker in workers:
				if no_tasks is not True and worker.is_free() and (max_model_num < 0 or model_id < max_model_num):
					msg = {"type":"req", "modelId":model_id, 'dataRange':'%d_%d' %(skip_rows, max_rows), 'max_rows':max_rows, 'skip_rows':skip_rows}
					worker.send(msg)
					model_id += 1
					skip_rows += max_rows

			if max_model_num > 0 and model_id >= max_model_num:
				no_tasks = True

			free_worker_num = 0
			for worker in workers:
				if worker.is_busy():
					msg = worker.recv(timeout=1)
					if msg is False:
						continue
					if msg["type"] == "noMoreData":
						no_tasks = True	
					else:	
						model_file = msg["modelFile"]
						models.append(model_file)

					free_worker_num += 1
				else:
					free_worker_num += 1

			if no_tasks is True and free_worker_num == worker_num:
				break

		for worker in workers:
			worker.stop()

		print "model trainning : %f seconds" %(time.time() -  start_time)

		for model in models:
			print "model file is %s" %(model)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "not enough arguments: feature"
		sys.exit(0)

	feature = sys.argv[1]

	lr_trainer = LrTrainer(feature=feature)
	lr_trainer.train_mp()
