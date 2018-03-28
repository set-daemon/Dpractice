#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# author: set_daemon@126.com
# date: 2018-03-12

import psutil
import os

def memory_usage(info):
	rss_total = psutil.Process(os.getpid()).memory_info().rss
	print u'[%s] RSS %fByte,%fKB,%fMB' %(info, mmount, mmount/1024.0, mmount/1024.0/1024)
