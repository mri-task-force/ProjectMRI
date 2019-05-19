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
import utility.logger

## setttings #############################################
server_8 = True  # 是否为8卡服务器
model_name = 'model'
class_specifier = {0: 0, 1: 1, 2: 2, 3: 3}      # 类别说明符, 便于做 2 ,3, 4 分类等

num_classes = 4

##########################################################

## datasets #################################################
# [39人小数据集, 651人CC_ROI, 363人6_ROI]
PATHS_datasets = [
    {
        'xlsx_path': '/home/share/Datasets/data/information.xlsx' if server_8 else '../../Datasets/data/information.xlsx',
        'sheet_name': 'Sheet1',  
        'data_path': '/home/share/Datasets/data/' if server_8 else '../../Datasets/data/',
    },
    {
        'xlsx_path': '/home/share/Datasets/2019_rect_pcr_data/information.xlsx' if server_8 else '../../Datasets/2019_rect_pcr_data/information.xlsx',
        'sheet_name': 0,
        'data_path': '/home/share/Datasets/2019_rect_pcr_data/CC_ROI/' if server_8 else '../../Datasets/2019_rect_pcr_data/CC_ROI/',
    },
    {
        'xlsx_path': '/home/share/Datasets/2019_rect_pcr_data/information.xlsx' if server_8 else '../../Datasets/2019_rect_pcr_data/information.xlsx',
        'sheet_name': 1,
        'data_path': '/home/share/Datasets/2019_rect_pcr_data/6_ROI/' if server_8 else '../../Datasets/2019_rect_pcr_data/6_ROI/',
    }
]

PATH_split_json = './process/split.json'    # 数据信息json路径
##########################################################

# folders
_DIR_trained_model = './trained_model/'
_DIR_logs = './log/'
_DIR_patient_result = './patient_result/'
_DIR_tblogs = './tblogs/'

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

PATH_model = '{}{}{}.pt'.format(_DIR_trained_model, now_time, model_name)     # to save the model
DIR_tblog = '{}{}{}/'.format(_DIR_tblogs, now_time, model_name)    # tensorboard log
PATH_log = '{}{}{}.log'.format(_DIR_logs, now_time, model_name)
PATHS_patient_result_json = [
    '{}{}train.json'.format(_DIR_patient_result, now_time),
    '{}{}test.json'.format(_DIR_patient_result, now_time)
]
DIR_tb_cm = DIR_tblog + 'cm/'

log = utility.logger.Logger(PATH_log, level='debug')