#!~/tf-py3/bin/python
# -*- coding: utf-8 -*-

"""
    1.运行例子
    python test_vgg19.py --path test_data  #如果不指定path默认就为test_data
"""

import numpy as np
import tensorflow as tf
import vgg19
import utils
import test_color as tcolor

import os
import argparse

import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

TEST_DATA="test_data"
TRAIN_DATA="20180330"

def writeImg(imgname,txt,bakpath="bak"):
  #font = cv2.FONT_HERSHEY_SIMPLEX
  #img = cv2.imread(imgname) 
  #cv2.putText(img,txt , (int(img.shape[0]/3 - len(txt)), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
  #cv2.imwrite(os.path.join(bakpath,imgname),img)

  # img = cv2.imdecode(np.fromfile(imgname,dtype=np.uint8),-1)
  # cv2.putText(img,txt , (int(img.shape[0]/3 - len(txt)), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
  # cv2.imencode('.jpg',img)[1].tofile(os.path.join(bakpath,imgname))

  img = Image.open(imgname)
  draw = ImageDraw.Draw(img) 
  font = ImageFont.truetype("font/simhei.ttf", 20, encoding="utf-8") 
  draw.text((0, 0), "eg：打印在这里", (0, 0, 255), font=font) 
  img.save(os.path.join(bakpath,imgname))

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

def loadFrom(path=TEST_DATA):
    imgs=[]
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png','PNG']
    for (root,dirs,files) in os.walk(path):
        for f in files:
            #print(os.path.join(root,f))
            for extension in extensions:
                if f.endswith(extension):
                    imgs.append(os.path.join(root,f))
        for d in dirs:
            pass#print(os.path.join(root,d))
    return imgs

 #"test-save.npy"
def tensor_imgdata(imgdata,images,vgg,bakpath):
    num = len(imgdata)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)
    with tf.device('/cpu:0'):  #'/cpu:0'
        with tf.Session() as sess:
            
            feed_dict = {images: batch}
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            #print(prob)
            for x in range(0,num):

                print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
                res = utils.print_prob(prob[x], './synset.txt')    

                #writeImg(imgdata[x],res[10:],bakpath)

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # img = cv2.imread(imgdata[x]) 
                # cv2.putText(img, res[10:], (int(img.shape[0]/3 - len(res[10:])), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
                # cv2.imwrite(os.path.join(bakpath,imgdata[x]),img)


def main():
    basepath = TEST_DATA
    bakpath  = "bak"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="读取某个文件夹下的所有图像文件,default="+TEST_DATA,default=TEST_DATA)
    parser.add_argument("--npy", default='./vgg19.npy')
    #parser.add_argument("--tpath", help="读取训练图像数据,default="+TRAIN_DATA,default=TRAIN_DATA)
    args = parser.parse_args()

    if args.path:
        basepath = args.path
    
    mkdir(bakpath+"/"+basepath)
    imgdata=[]
    imgdata = loadFrom(basepath) #20180330
    num=len(imgdata)
    if num == 0:
        utils.printcolor("图像文件数量为0",mode='bold',fore='red')
        return

    per=10 if (num>10) else num

    count =  int(num/per) if (num % per == 0) else int(num/per)+1
    print(per,num,count,num % per)

    vgg = vgg19.Vgg19(args.npy)
    images = tf.placeholder("float", [per, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    for x in range(0,count):
        xdata=imgdata[x*per:x*per+per]
        #print(len(xdata))
        if len(xdata) == num % per:
            vggx = vgg19.Vgg19(args.npy)
            images = tf.placeholder("float", [len(xdata), 224, 224, 3])
            with tf.name_scope("content_vgg"):
                vggx.build(images)
            tensor_imgdata(xdata,images,vggx,bakpath)
        else:
            tensor_imgdata(xdata,images,vgg,bakpath)    

def train():
    print("训练数据")


def test():
    res = "n01440764 tench, Tinca tinca"
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread("./test_data/tiger.jpeg") 
    print(img.shape,len(res[10:]))
    cv2.putText(img, res[10:], (int(img.shape[0]/3 - len(res[10:]) *1), int(img.shape[1]/2)), font, 0.8, (0, 255, 0), 2)
    cv2.imwrite("./test_data/tiger2.jpeg",img)    

if __name__ == "__main__":
    main()
    #test()

