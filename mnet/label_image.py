# -*- coding: utf-8 -*-

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os
import cv2

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import tensorflow as tf

import test_color as tcolor

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

image_dir = 'imgdata/coco-animals/val/'
def loadFrom(path=image_dir):
    imgs=[]
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png','PNG']
    for (root,dirs,files) in os.walk(path):
        for f in files:
            for extension in extensions:
                if f.endswith(extension):
                    imgs.append(os.path.join(root,f))
        for d in dirs:
            pass
    return imgs

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

def writeImg(imgname,txt,bakpath="bak"):
  #font = cv2.FONT_HERSHEY_SIMPLEX
  #img = cv2.imread(imgname) 
  #cv2.putText(img,txt , (int(img.shape[0]/3 - len(txt)), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
  #cv2.imwrite(os.path.join(bakpath,imgname),img)

  # img = cv2.imdecode(np.fromfile(imgname,dtype=np.uint8),-1)
  # cv2.putText(img,txt , (int(img.shape[0]/3 - len(txt)), int(img.shape[1]/2)), font, 1, (0, 255, 0), 2)
  # cv2.imencode('.jpg',img)[1].tofile(os.path.join(bakpath,imgname))

  img  = Image.open(imgname)
  draw = ImageDraw.Draw(img) 
  font = ImageFont.truetype("font/simhei.ttf", 20, encoding="utf-8") 
  offset=(0,0)#(int(img.width/3 - len(txt)), int(img.height/2))
  draw.text(offset, " 奥爱你打印在这里", (0, 255, 0), font=font) 
  img.save(os.path.join(bakpath,imgname))
  

if __name__ == "__main__":
  file_name = "rose.jpeg"
  model_file = "tmp/output_graph.pb"   
  label_file = "tmp/output_labels.txt" 
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  
  xmode = 0
  if xmode == 1:
    model_file = "tmp/mnet/mobilenet_v1_1.0_224/frozen_graph.pb" 
    label_file = "tmp/mnet/mobilenet_v1_1.0_224/labels.txt" 
    input_mean = 127.5
    input_std  = 127.5
    input_layer = "input"
    output_layer = "MobilenetV1/Predictions/Reshape"
  
  

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)


  imgdata = []
  imgdata = loadFrom()

  #mkdir("bak/"+image_dir)
  #writeImg(imgdata[0],"你好")
  #imgdata = []
  #num=len(imgdata)

  for fname in imgdata:
    t = read_tensor_from_image_file(fname,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: t})
      end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-1:][::-1]
    labels = load_labels(label_file)
    print('\nEvaluation time (1-image): {:.3f}s'.format(end-start))

    for i in top_k:
      #ret+=(labels[i]+" "+results[i]+",")
      label=tcolor.UseStyle(labels[i],mode = 'bold',fore = 'green')
      print("["+fname+"]", label, results[i])
