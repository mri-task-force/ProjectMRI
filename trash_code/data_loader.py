import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import cv2

IMG_DIR = r"D:\Datasets\data\sub001\MRI\T2"
AN_IMG = r"D:\Datasets\data\sub005\MRI\T2\IMG-0001-00001.dcm"
ROW = 10
COL = 3

def read_images(img_dir):
    images_3d = []
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        file_type = os.path.splitext(img_name)[1]

        if (not(file_type == '.dcm')):
            print("File is not .dcm")
            continue
        
        img_path = os.path.join(img_dir, img_name)
        if os.path.isfile(img_path):
            # print(img_path)
            sitk_image = sitk.ReadImage(img_path)
            np_image = sitk.GetArrayFromImage(sitk_image)
            images_3d.append(np_image[0])
    return np.array(images_3d)







def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)    
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName.decode('gb18030')
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex.encode('ISO-8859-1').decode('gb18030')
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    # information['NumberOfFrames'] = ds.NumberOfFrames    
    return information


def writeVideo(img_array):
    frame_num, width, height = img_array.shape
    # filename_output = filename.split('.')[0] + '.avi'    
    filename_output = "myvideo.avi"
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))    
    for img in img_array:
        video.write(img)
    video.release()


# 显示一个系列图
def show_imgs(np_imgs):
    for j in range(COL):
        for i in range(ROW):
            index = ROW * j + i
            if (index >= np_imgs.shape[0]):
                break
            plt.subplot(COL, ROW, ROW * j + i + 1)
            plt.imshow(np_imgs[index,:,:], cmap='gray')
            # print(index)
        
    plt.show()


# np_imgs = read_images(IMG_DIR)
# print(np_imgs.shape)
# show_imgs(np_imgs)
# print(loadFileInformation(AN_IMG))
