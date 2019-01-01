# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:00:48 2019

@author: GEAR
"""
import time
import pandas as pd
import numpy as np


root = 'D:/data/'      # 文件太大没法git

##########################--pandas读取超大文件--################################
#train = pd.read_csv(root + 'security_train.csv',iterator=True)
#loop = True
#chunksize = 10000
#chunks = []
#
#while loop:   
#    try:
#        chunk = train.get_chunk(10000)
#        chunks.append(chunk)
#    except StopIteration:
#        loop = False
#        print('Iteration is stopped.')
#
#train = pd.concat(chunks, ignore_index=True)

def ReadBigData(reader, chunksize):
    '''
    Parameter:
        reader: pandas定义的迭代器
        chunksize: 每次读取的文件最大行数
    Returns:
        data: 最终读取的数据
    '''
    chunks = []
    loop = True
    t1 = time.time()
    while loop:
        try:
            chunk = reader.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iternation is stopped.')
    t2 = time.time()
    print('read times %.3f' % (t2-t1))   
    data = pd.concat(chunks, ignore_index=True)
    
    return data



##############################--end--##########################################

reader = pd.read_csv(root + 'security_train.csv',iterator=True)
train = ReadBigData(reader, chunksize=10000)



