#!~/tf-py3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import vgg16
import utils
import cv2
import os

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

def loadFrom(path="test_data"):
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

def tensor_imgdata(imgdata,images,vgg,bakpath):
    num = len(imgdata)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            
            feed_dict = {images: batch}
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            #print(prob)
            for x in range(0,num):
                print("-"*25,imgdata[x],"-"*25)
                res = utils.print_prob(prob[x], './synset.txt')    

                #print(res[10:])
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.imread(imgdata[x]) 
                cv2.putText(img, res[10:], (int(img.shape[0]/3 - len(res[10:])), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(bakpath,imgdata[x]),img)

def main():
    basepath = "test_data"
    bakpath  = "bakvgg16"
    mkdir(bakpath+"/"+basepath)
    imgdata=[
        "./test_data/tiger.jpeg"
    ]
    imgdata = loadFrom(basepath) #20180330
    num=len(imgdata)
    if num == 0:
        utils.printcolor("图像文件数量为0",mode='bold',fore='red')
        return

    per=10 if (num>10) else num
    count =  int(num/per) if (num % per == 0) else int(num/per)+1
    print(per,num,count,num % per)

    vgg = vgg16.Vgg16()
    images = tf.placeholder("float", [per, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    for x in range(0,count):
        xdata=imgdata[x*per:x*per+per]
        #print(len(xdata))
        if len(xdata) == num % per:
            vggx = vgg16.Vgg16()
            images = tf.placeholder("float", [len(xdata), 224, 224, 3])
            with tf.name_scope("content_vgg"):
                vggx.build(images)
            tensor_imgdata(xdata,images,vggx,bakpath)
        else:
            tensor_imgdata(xdata,images,vgg,bakpath) 

if __name__ == "__main__":
    main()

# imgdata=[
#     "./test_data/m1.jpeg"
# ]
# num=len(imgdata)
# batchs=[]
# for x in imgdata:
#     img= utils.load_image(x)
#     b = img.reshape((1, 224, 224, 3))
#     batchs.append(b)

# batch = np.concatenate(batchs, 0)

# with tf.device('/cpu:0'):
#     with tf.Session() as sess:
#         images = tf.placeholder("float", [num, 224, 224, 3])
#         feed_dict = {images: batch}

#         vgg = vgg16.Vgg16()
#         with tf.name_scope("content_vgg"):
#             vgg.build(images)

#         prob = sess.run(vgg.prob, feed_dict=feed_dict)
#         #print(prob)
#         for x in range(0,num):
#             print("-"*25,imgdata[x],"-"*25)
#             utils.print_prob(prob[x], './synset.txt')


