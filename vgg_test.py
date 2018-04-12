import tensorflow as tf
from vgg.vgg19 import Vgg19
from vgg.vgg16 import Vgg16
import matplotlib.pyplot as plt
import utils
import test_color as tcolor
import numpy as np

# def test_image(path_image, num_class,npypath=train_npy_path):
#     class_name = ['not fire', 'fire']
#     img_string = tf.read_file(path_image)
#     img_decoded = tf.image.decode_png(img_string, channels=3)
#     img_resized = tf.image.resize_images(img_decoded, [224, 224])
#     img_resized = tf.reshape(img_resized, shape=[1, 224, 224, 3])
 
#     model = vgg19.Vgg19(bgr_image=img_resized, num_class=num_class, vgg19_npy_path=npypath)
#     score = model.fc8

#     prediction = tf.argmax(score, 1)
#     #saver = tf.train.Saver()
#     saver = tf.train.import_meta_graph('checkpoints/vgg_16_train.ckpt.meta')
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         #saver.restore(sess, "checkpoints/vgg_16_train.ckpt")
#         saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
#         prob = sess.run(prediction)
#         print("prob=",prob)
#         plt.imshow(img_decoded.eval())
#         plt.title("Class:" + class_name[prob[0]])
#         plt.show()

def test_image_vgg19(imgdata,npypath,xmode=0):
    num = len(imgdata)
    images = tf.placeholder("float", [num, 224, 224, 3])
    model = Vgg19(npypath)
    with tf.name_scope("content_vgg"):#content_vgg
        model.build(images,xmode)

    # score = model.fc8
    # prediction = tf.argmax(score, 1)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)
    #saver = tf.train.import_meta_graph('checkpoints/vgg_16_train.ckpt.meta')
    with tf.device('/cpu:0'): 
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())     
            #saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
            feed_dict = {images: batch}
            prob = sess.run(model.prob, feed_dict=feed_dict)
            for x in range(0,num):
                print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
                res = utils.print_prob(prob[x], './synset.txt')   

            #print("prediction:",sess.run(prediction)) 
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # img = cv2.imread(imgdata[x]) 
                # cv2.putText(img, res[10:], (int(img.shape[0]/3 - len(res[10:])), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
                # cv2.imwrite(os.path.join(bakpath,imgdata[x]),img)

def test_image_vgg16(imgdata,npypath,xmode=0):
    num = len(imgdata)
    images = tf.placeholder("float", [num, 224, 224, 3])
    model = Vgg16(npypath)
    with tf.name_scope("content_vgg"):#content_vgg
        model.build(images,xmode)
    batchs=[]
    for x in imgdata:
        img= utils.load_image(x)
        b = img.reshape((1, 224, 224, 3))
        batchs.append(b)
    batch = np.concatenate(batchs, 0)

    with tf.device('/cpu:0'): 
        with tf.Session() as sess:
            feed_dict = {images: batch}
            prob = sess.run(model.prob, feed_dict=feed_dict)
            for x in range(0,num):
                print("-"*25, tcolor.UseStyle(imgdata[x],mode = 'bold',fore = 'white') ,"-"*25)
                res = utils.print_prob(prob[x], './synset.txt')   





def ckpt2npy(ckpt_path,savepath):
    from tensorflow.python import pywrap_tensorflow
    #checkpoint_path='checkpoints/vgg_16_train.ckpt'#your ckpt path
    reader=pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map=reader.get_variable_to_shape_map()
    vgg19={}
    for key in var_to_shape_map:
        sStr_2=key #key[:-2]
        print ("tensor_name",key,sStr_2)
        if not sStr_2 in vgg19:
            vgg19[sStr_2]=[reader.get_tensor(key)]
        else:
            vgg19[sStr_2].append(reader.get_tensor(key))
    np.save(savepath,vgg19)

if __name__ == '__main__':
    npy_path_vgg19 = 'vgg/vgg19.npy'
    npy_path_vgg16 = 'vgg/vgg16.npy'

    #train_npy_path_vgg19 = 'vgg/ckpt_vgg19.npy'
    #train_npy_path_vgg16 = 'vgg/ckpt_vgg16.npy'
    imgdata = ['imgdata/test_data/tiger.jpeg']

    #test_image_vgg19(imgdata,npy_path_vgg19,0)
    #test_image_vgg19(imgdata,train_npy_path_vgg19,1)
    test_image_vgg16(imgdata,npy_path_vgg16)
    
    #ckpt2npy('vgg/vgg_16.ckpt','vgg/ckpt_vgg16.npy')
    #ckpt2npy('vgg/vgg_19.ckpt','vgg/ckpt_vgg19.npy')


    #test_image('./test_data/tiger.jpeg', 2)

