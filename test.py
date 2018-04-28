import cv2
import numpy as np
from os.path import join
from doc_file_utils import *

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
## 1.a setup BOW
bow_train = cv2.BOWKMeansTrainer(8)  # toy world, you want more.
bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)

## 1.b add positives and negatives to the bowtrainer, use SIFT DescriptorExtractor
def feature_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


basepath = "./images/_panda"
images = getDanhSachFiles('_panda')
for i in range(len(images)):
    bow_train.add(feature_sift(basepath+"/"+images[i]))

bp = "./images/_airplanes"
images2= getDanhSachFiles('_airplanes')
for i in range(len(images2)):
    bow_train.add(feature_sift(bp + "/" + images2[i]))

## 1.c kmeans cluster descriptors to vocabulary
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)
print("bow vocab", np.shape(voc), voc)


## 2.a gather svm training data, use BOWImgDescriptorExtractor
def feature_bow(fn):
    im = cv2.imread(fn, 0)
    return bow_extract.compute(im, detect.detect(im))


traindata, trainlabels = [], []

for i in range(len(images)):
    traindata.extend(feature_bow(basepath+"/"+images[i]))
    trainlabels.append(999)
for i in range(len(images2)):
    traindata.extend(feature_bow(bp+"/"+images2[i]))
    trainlabels.append(888)

print("svm items", len(traindata), len(traindata[0]))

## 2.b create & train the svm
svm = cv2.ml.SVM_create()
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))


## 2.c predict the remaining 2*2 images, use BOWImgDescriptorExtractor again
def predict(fn):
    f = feature_bow(fn);
    p = svm.predict(f)
    print(p)


predict("C:\\Users\\quocb14005xx\\PycharmProjects\\CV_PYTHON\\baocao-tgmt\\images\\_panda\\image_0002.jpg")
