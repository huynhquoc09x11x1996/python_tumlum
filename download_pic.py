import requests
import urllib.request
import time
import cv2

link_can_down='http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07739125'

#GET LINKS
contents=requests.get(link_can_down)
links=contents.text.split("\n")


def get_milis():
    millis = int(round(time.time() * 1000))
    return millis

#DOWN BY LINK
def downloader(image_url):
    file_name = get_milis()
    full_file_name = "C:\\Users\\quocb14005xx\\PycharmProjects\\CV_PYTHON\\baocao-tgmt\\images\\1-tao\\"+str(file_name) + '.jpg'
    r=urllib.request.urlretrieve(image_url,full_file_name)
    return r
for i in range(1,1200):
    try:
        downloader(links[i])
        print(i)
    except:
        continue

