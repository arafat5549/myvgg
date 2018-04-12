#!~/tf-py3/bin/python
# -*- coding: utf-8 -*- 


"""
    vgg19训练精度
    注意查看train后的精确度，应该会有小范围提升
"""
import numpy as np
import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import test_color as tcolor
import os
import cv2
import argparse

TEST_DATA="test_data"
TRAIN_DATA="20180330"

def true_result(idx):
    return [1 if i == idx else 0 for i in range(1000)]

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



def tensor_imgdata(sess,imgdata,images,train_mode,results,vgg,bakpath):
    num = len(imgdata)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)
    # with tf.device('/cpu:0') as device:
    #     with tf.Session() as sess:
    
    feed_dict = {images: batch, train_mode: False}
    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    for x in range(0,num):
        print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
        res = utils.print_prob(prob[x], './synset.txt')
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # img = cv2.imread(imgdata[x]) 
        # cv2.putText(img, res[10:], (int(img.shape[0]/3 - len(res[10:])), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(bakpath,imgdata[x]),img)
    
    #print(tcolor.UseStyle("-"*125,mode = 'bold',fore = 'red'))
    true_out = tf.placeholder(tf.float32, [num, 1000])
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)   
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch, true_out: results, train_mode: True})
    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    for x in range(0,num):
        print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
        utils.print_prob(prob[x], './synset.txt',fore='blue') 
    


def main():
    basepath  = TEST_DATA
    trainpath = TRAIN_DATA
    bakpath  = "bak"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="读取某个文件夹下的所有图像文件,default="+TEST_DATA,default=TEST_DATA)
    parser.add_argument("--tpath", help="读取训练图像数据,default="+TRAIN_DATA,default=TRAIN_DATA)
    args = parser.parse_args()

    if args.path:   basepath = args.path
    if args.tpath:  trainpath = args.tpath
    
    mkdir(bakpath+"/"+basepath)
    imgdata=[]
    imgdata = loadFrom(basepath) #20180330
    num=len(imgdata)
    if num == 0:
        utils.printcolor("图像文件数量为0",mode='bold',fore='red')
        return

    per=5 if (num>5) else num
    count =  int(num/per) if (num % per == 0) else int(num/per)+1
    print(per,num,count,num % per)


    with tf.device('/cpu:0') as device:
        sess = tf.Session()

        vgg = vgg19.Vgg19('./vgg19.npy')
        images = tf.placeholder("float", [per, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg.build(images,train_mode)
        print(vgg.get_var_count())
        sess.run(tf.global_variables_initializer())

        results=[]
        for i in range(0,num):
            results.append(true_result(728))

        for x in range(0,count):
            xdata=imgdata[x*per:x*per+per]
            result=results[x*per:x*per+per]
            if len(xdata) == num % per:
                xdata=imgdata[-per:]
                result=results[-per:]
                tensor_imgdata(sess,xdata,images,train_mode,result,vgg,bakpath)
            else:
                tensor_imgdata(sess,xdata,images,train_mode,result,vgg,bakpath) 
        vgg.save_npy(sess, './vgg19_train.npy')



'''
def train():
    imgdata=[
        "./test_data/plastic.png",
        "./test_data/tiger.jpeg"
    ]
    num=len(imgdata)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)

    results=[
        true_result(728),true_result(292)
    ]
    with tf.device('/cpu:0'):
        sess = tf.Session()

        images = tf.placeholder(tf.float32, [num, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [num, 1000])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build(images, train_mode)
        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())
        sess.run(tf.global_variables_initializer())

        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
        for x in range(0,num):
            print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'normal',fore = 'white') ,"-"*25)
            utils.print_prob(prob[x], './synset.txt')

        # simple 1-step training
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        sess.run(train, feed_dict={images: batch, true_out: results, train_mode: True})
        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
        for x in range(0,num):
            print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
            utils.print_prob(prob[x], './synset.txt',fore='blue')
        # test save
        vgg.save_npy(sess, './vgg19_train.npy')
'''

if __name__ == "__main__":
    #train()
    main()
