import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

model_path=''

def choose_image:
    global img_path
    img_path=''

def choose_model:
    global model_path
    if model_flag==0:
        model_path=''
    if model_flag==1:
        model_path=''
    if model_flag==2:
        model_path=''

def create_session():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path+'/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
        sess = tf.compat.v1.Session(graph=detection_graph)

    return (detection_graph,sess)

def img_preprocess(reshape=(320,320)):

    im=cv2.imread(img_path)
    im=cv2.resize(im,(reshape[0],reshape[1]))
    img=np.expand_dims(im,axis=0)

    return img

def detect(detection_graph,sess,img):
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: img})

    return (boxes,scores,classes,num)

def visualize(img,boxes,scores,classes,num,threshold=0.5):

    im1=np.squeeze(img)
    for i in range(len(int(num[0]))):
        if scores[0][i]>threshold:
            y0=int(boxes[0][i][0]*im.shape[0])
            x0=int(boxes[0][i][1]*im.shape[1])
            y1=int(boxes[0][i][2]*im.shape[0])
            x1=int(boxes[0][i][3]*im.shape[1])
            cv2.rectangle(im1,(x0,y0),(x1,y1),(255,0,0),2)
            cv2.putText(im1,'class_id:'+str(int(classes[0][i])),(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            #plt.imshow(im1)

    return im1
            
    

    
