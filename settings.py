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
import json

# import the files of mine
import utility.logger

## setttings #############################################
new_init = False    # 是否重新初始化数据集(生成split.json)
server_8 = True  # 是否为8卡服务器
model_name = 'model'
class_specifier = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}      # 类别说明符, 便于做 2 ,3, 4 分类等

test_folds = [0]
train_folds = [x for x in range(5) if x not in test_folds]
is_from_split = True    # 若为True, 则按以下split划分数据集; 否则重新随机划分
split = {
    "0": ["sub272", "sub306", "sub118", "sub292", "sub320", "sub156", "sub054", "sub217", "sub241", "sub311", "sub099", "sub269", "sub080", "sub105", "sub203", "sub196", "sub029", "sub271", "sub095", "sub082", "sub287", "sub109", "sub190", "sub288", "sub145", "sub013", "sub244", "sub061", "sub180", "sub018", "sub106", "sub296", "sub154", "sub123", "sub185", "sub143", "sub067", "sub183", "sub254", "sub290", "sub177", "sub121", "sub207", "sub253", "sub137", "sub200", "sub131", "sub100", "sub128", "sub120", "sub014", "sub307", "sub259", "sub318", "sub024", "sub142", "sub260", "sub107", "sub266", "sub363", "sub108", "sub060", "sub065", "sub304", "sub027", "sub164", "sub239", "sub076"],
    "1": ["sub337", "sub157", "sub236", "sub015", "sub087", "sub284", "sub008", "sub205", "sub167", "sub025", "sub351", "sub144", "sub073", "sub070", "sub132", "sub165", "sub227", "sub291", "sub211", "sub041", "sub279", "sub188", "sub313", "sub005", "sub163", "sub086", "sub319", "sub110", "sub274", "sub194", "sub047", "sub226", "sub032", "sub187", "sub002", "sub225", "sub230", "sub182", "sub209", "sub246", "sub074", "sub001", "sub149", "sub342", "sub361", "sub035", "sub348", "sub303", "sub283", "sub189", "sub007", "sub173", "sub152", "sub119", "sub051", "sub171", "sub295", "sub352", "sub133", "sub081", "sub223", "sub162", "sub093", "sub056", "sub170", "sub064", "sub258"], 
    "2": ["sub020", "sub346", "sub130", "sub006", "sub218", "sub305", "sub003", "sub336", "sub362", "sub042", "sub242", "sub186", "sub221", "sub273", "sub310", "sub229", "sub046", "sub354", "sub084", "sub359", "sub278", "sub179", "sub112", "sub031", "sub113", "sub178", "sub334", "sub222", "sub055", "sub097", "sub240", "sub169", "sub356", "sub021", "sub092", "sub338", "sub289", "sub057", "sub096", "sub249", "sub012", "sub248", "sub316", "sub234", "sub126", "sub233", "sub312", "sub224", "sub036", "sub026", "sub344", "sub124", "sub010", "sub349", "sub261", "sub044", "sub184", "sub115", "sub135", "sub083", "sub172", "sub298", "sub114", "sub238", "sub302", "sub255"],
    "3": ["sub072", "sub257", "sub358", "sub028", "sub071", "sub210", "sub019", "sub146", "sub335", "sub141", "sub101", "sub251", "sub066", "sub214", "sub208", "sub011", "sub277", "sub181", "sub192", "sub022", "sub050", "sub016", "sub276", "sub270", "sub069", "sub341", "sub175", "sub098", "sub360", "sub191", "sub197", "sub195", "sub122", "sub174", "sub017", "sub116", "sub139", "sub314", "sub030", "sub293", "sub262", "sub355", "sub153", "sub148", "sub023", "sub089", "sub300", "sub333", "sub117", "sub053", "sub357", "sub280", "sub045", "sub077", "sub340", "sub219", "sub033", "sub085", "sub364", "sub353", "sub052", "sub294", "sub247", "sub091", "sub339", "sub308", "sub235"], 
    "4": ["sub068", "sub250", "sub078", "sub281", "sub039", "sub138", "sub158", "sub150", "sub136", "sub034", "sub252", "sub286", "sub232", "sub166", "sub168", "sub267", "sub199", "sub245", "sub048", "sub263", "sub176", "sub265", "sub004", "sub285", "sub049", "sub009", "sub088", "sub037", "sub129", "sub111", "sub059", "sub063", "sub202", "sub231", "sub317", "sub243", "sub079", "sub215", "sub075", "sub147", "sub204", "sub140", "sub198", "sub228", "sub090", "sub347", "sub062", "sub102", "sub134", "sub151", "sub040", "sub345", "sub038", "sub301", "sub343", "sub350", "sub282", "sub237", "sub094", "sub161", "sub160", "sub103", "sub256", "sub299", "sub297", "sub155", "sub058", "sub201", "sub315", "sub104"]
}

# with open('./foo.json', 'w') as json_file:
#     json_file.write(json.dumps(split))   # 写入json

# print(split)
# print(len(split.keys()))

# for key, val in split.items():
#     print(key, val)

##########################################################

num_classes = len(set(class_specifier.values()))

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