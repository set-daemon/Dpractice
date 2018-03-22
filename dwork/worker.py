#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-13
# env:  under python2.7

import sys
import time

import multiprocessing

class Worker(object):
	STATE_NORMAL = 1
	STATE_FREE = 2
	STATE_BUSY = 3
	STATE_DEAD = 4
	def __init__(self, name, cb, ctx):
		self.name = name
		self.cb = cb
		self.ctx = ctx
		self.status = Worker.STATE_NORMAL 

	@staticmethod
	def suitable_worker_num():
		return multiprocessing.cpu_count()

	def start(self):
		self.child_pipe, self.parent_pipe = multiprocessing.Pipe()
		self.handle = multiprocessing.Process(target = Worker.process_cb, name = self.name, args=(self.name, self.child_pipe, self.cb, self.ctx))

		self.handle.start()
		self.child_pipe.close()

		self.status = Worker.STATE_FREE

		return True

	def stop(self):
		msg = {'type':'offline'}
		self.parent_pipe.send(msg)
		self.status = Worker.STATE_DEAD

	# only for parent process
	def send(self, msg):
		if self.status != Worker.STATE_FREE:
			return False

		self.status = Worker.STATE_BUSY
		self.parent_pipe.send(msg)

		return True

	def recv(self, timeout=0.01):
		if self.status != Worker.STATE_BUSY:
			return False

		if self.parent_pipe.poll(timeout) != True:
			return False

		self.status = Worker.STATE_FREE
		return self.parent_pipe.recv()

	def is_free(self):
		return self.status == Worker.STATE_FREE

	def is_busy(self):
		return self.status == Worker.STATE_BUSY

	@staticmethod
	def process_cb(worker_name, pipe, cb, ctx):

		while True:
			if pipe.poll(0.01) != True:
				time.sleep(0.1)
				continue

			msg = pipe.recv()

			if msg["type"] == "offline":
				break

			# if cb works with a long period, it will cause the recv EOFError
			rsp = cb(msg, ctx)
			pipe.send(rsp)

		print "Worker %s quit" %(worker_name)

def test_cb(msg, ctx):
	# the following codes makes the callback function persist for a long time
	#count = 0
	#for i in range(0, 0xFFFFFFF):
	#	count = 1

	rsp = {"type":"rsp", "task_id":msg["task_id"], "name": ctx["name"]}

	return rsp

if __name__ == "__main__":

	ctx = {"name":"your name"}
	worker_num = Worker.suitable_worker_num()
	workers = []

	for i in range(0, worker_num):
		worker = Worker('testWorker %d' %(i), test_cb, ctx)
		workers.append(worker)
		worker.start()

	max_task = 10
	task_id = 1
	while task_id < max_task:
		for worker in workers:
			if worker.is_free():
				msg = {'type':'req', 'task_id':task_id}
				worker.send(msg)
				task_id += 1
			else:
				rsp = worker.recv()
				print rsp
		print "sleep 2 seconds"
		time.sleep(2)

	print "task completed"
	for worker in workers:
		worker.stop()

	time.sleep(1)
