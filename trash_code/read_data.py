#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utility.py
@Time    :   2019/04/05 00:41:37
@Author  :   Wu
@Version :   1.0
@Desc    :   Some utility functions
'''

import os
import SimpleITK as sitk
import numpy as np
import torch

# import the files of mine
import utility.evaluation
import models
from logger import log


def get_image_paths(img_dir):
    """
    return a `list`, the paths of the images (slides) in `img_dir`\\
    Args: \\
        img_dir: the directory of the images (slides)\\
    """
    # if there is no such directory, return an empty list. 
    # Notice: Some directories are in the excel file, while not exist in the dataset folder.
    if not os.path.exists(img_dir): 
        return []
    paths = []
    for img_name in os.listdir(img_dir):
        # check the file type
        file_type = os.path.splitext(img_name)[1]
        if (not(file_type == '.dcm')):
            log.logger.warning("File is not .dcm")
            continue
        paths.append(os.path.join(img_dir, img_name))
    return sorted(paths)


def read_image(img_path):
    """
    read an image from `img_path` and return its numpy image\\
    Args: \\
        img_path: the path of the image\\
    """
    sitk_image = sitk.ReadImage(img_path)
    np_image = sitk.GetArrayFromImage(sitk_image)
    return np_image[0].astype(np.float32)