# -*- coding: utf-8 -*-

import tensorflow as tf
import utils
import test_color as tcolor
import numpy as np

import vgg

TRAIN_PATH="imgdata/20180330"

CKPT_VGG19 = "vgg/vgg_19.ckpt"

TRAIN_TYPE = 0 

'''
	1.根据一句训练好的npy划分 污染物类型 将结果存入txt文件
'''
def shuffle():

	pass

'''
    2.根据分类好的图像集合 进行分类处理
'''
def classify():
	pass


if __name__ == '__main__':
	# print("-"*150)
	# imgdata = ['imgdata/test_data/tiger.jpeg']
	# num = len(imgdata)
	# inputs = tf.placeholder("float", [num, 224, 224, 3])
	# model,endpoints = vgg.vgg_e(inputs,
 #           num_classes=1000,
 #           is_training=True,
 #           dropout_keep_prob=0.5,
 #           spatial_squeeze=True,
 #           scope='vgg_19',
 #           fc_conv_padding='VALID',
 #           global_pool=False)

	# #print("-----")
	# batchs=[]
	# for x in imgdata:
	# 	img= utils.load_image(x)
	# 	b = img.reshape((1, 224, 224, 3))
	# 	batchs.append(b)
	# batch = np.concatenate(batchs, 0)

	# with tf.Session() as sess:
	# 	feed_dict = {inputs: batch}
	# 	prob = sess.run(model.prob, feed_dict=feed_dict)
	# 	for x in range(0,num):
	# 		print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
	# 		utils.print_prob(prob[x], './synset.txt')  

 

