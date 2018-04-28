from svmutil import *
import cv2
import numpy as np
from doc_file_utils import *


#load file name
try:
    list_path = getListAbsolutePath_ofFolder('_panda')
except:
    print("Kiem tra lai folder name")

BoWs=cv2.BOWKMeansTrainer(10)

# trich dac trung cac anh cua folder name
for i in range(len(list_path)):
    print("trich xuat dac trung sift image "+ str(i+1))
    img = cv2.imread(list_path[i], cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    BoWs.add(descriptors)
print("trich xuat SIFT thanh cong!")

#tao tu dien
dictionary = BoWs.cluster()
print(dictionary.shape)

dict_arr=np.array(dictionary)




















# https://stackoverflow.com/questions/31414782/image-classification-in-opencv-python-based-on-training-set























# img = cv2.imread("the_book_thief.jpg", cv2.IMREAD_GRAYSCALE)  # img2 = cv2.imread("the_book_thief.jpg", cv2.IMREAD_GRAYSCALE)
#
# sift=cv2.xfeatures2d.SIFT_create()
#
#
# #111111111111
# # orb = cv2.ORB_create(nfeatures=1500)
# # keypoints, descriptors = orb.detectAndCompute(img, None)
# # img = cv2.drawKeypoints(img, keypoints, None)
# # cv2.imshow("Image1", img)
#
# #222222222
# kp ,desc= sift.detectAndCompute(img2, None)
# img2 = cv2.drawKeypoints(img2, kp, None)
#
#
#
