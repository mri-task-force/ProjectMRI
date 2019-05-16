#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   settings.py
@Time    :   2019/05/16 11:26:17
@Author  :   Wu
@Version :   1.0
@Desc    :   None
'''

import datetime
import os

# import the files of mine
import logger

## setttings #############################################
model_name = 'resnet34'

# folders
_DIR_trained_model = './trained_model/'
_DIR_logs = './log/'
_DIR_patient_result = './patient_result/'
_DIR_tblogs = './tblogs/'

##########################################################

# create folders
if not os.path.exists(_DIR_trained_model):
    os.makedirs(_DIR_trained_model)
if not os.path.exists(_DIR_logs):
    os.makedirs(_DIR_logs)
if not os.path.exists(_DIR_patient_result):
    os.makedirs(_DIR_patient_result)
if not os.path.exists(_DIR_tblogs):
    os.makedirs(_DIR_tblogs)

now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S=')   # 用于以下文件名的命名

# is_aug = True
# model_name = '{}Model-spc-cut-aug'.format(now_time) if is_aug else '{}Model-spc-cut'.format(now_time) # to save the model
# patient_json_dir = ['{}{}patient-train-spc-cut.json'.format(PATH_patient_result, now_time), '{}{}patient-test-spc-cut.json'.format(PATH_patient_result, now_time)] 
# if is_aug: 
#     patient_json_dir = ['{}{}patient-train-spc-cut-aug.json'.format(PATH_patient_result, now_time), '{}{}patient-test-spc-cut-aug.json'.format(PATH_patient_result, now_time)]
# tensorboard_dir = '{}{}tblog-spc-cut-aug'.format(PATH_tblogs, now_time) if is_aug else '{}{}tblog-spc-cut'.format(PATH_tblogs, now_time)
# logger_file_path = '{}{}{}.log'.format(PATH_log, now_time, 'logger-spc-cut-aug' if is_aug else 'logger-spc-cut')


PATH_model = '{}{}{}.pt'.format(_DIR_trained_model, now_time, model_name)     # to save the model
DIR_tblog = '{}{}{}/'.format(_DIR_tblogs, now_time, model_name)    # tensorboard log
PATH_log = '{}{}{}.log'.format(_DIR_logs, now_time, model_name)
PATHS_patient_result_json = [
    '{}{}train.json'.format(_DIR_patient_result, now_time),
    '{}{}test.json'.format(_DIR_patient_result, now_time)
]

log = logger.Logger(PATH_log, level='debug')