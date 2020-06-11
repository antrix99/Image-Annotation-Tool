import numpy as np
import cv2
import os
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.togglebutton import ToggleButton

desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

labels=['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
        'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
        'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush',
        'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush',
        'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other',
        'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other',
        'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill',
        'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss',
        'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow',
        'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad',
        'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
        'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
        'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree',
        'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood']

class BoxLayoutDemo(App):
    
    def build(self):
        
        self.filter_classes=list()
        
        self.threshold=0.5
        
        self.model_path=desktop_path+'\\Resolute_AI\\ssd_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'

        superBox        = BoxLayout(orientation='horizontal')

        verticalBox   = BoxLayout(orientation='vertical',size_hint=(0.1,1),spacing=5)

        button1         = Button(text='Open\nFolder',font_size='12sp')
        button1.bind(on_press=self.open_folder)

        button2         = Button(text="Next\nImage",font_size='12sp')
        button2.bind(on_press=self.display_next_image)

        button3         = Button(text="Previous\nImage",font_size='12sp')
        button3.bind(on_press=self.display_prev_image)

        button4         = Button(text="Save\nAnnotation",font_size='12sp')
        button4.bind(on_press=self.save_annotation)

        verticalBox.add_widget(button1)

        verticalBox.add_widget(button2)

        verticalBox.add_widget(button3)

        verticalBox.add_widget(button4)

        verticalBox0    = BoxLayout(orientation='vertical',size_hint=(0.8,1))


        self.file_chooser = FileChooserListView(size_hint=(1,0.5),dirselect=True,
                                                path=desktop_path)

        self.imageBox     = Image(size_hint=(1,0.5),width=500,height=300,
                             allow_stretch=True,source='img_back.jpg')

        verticalBox0.add_widget(self.file_chooser)

        verticalBox0.add_widget(self.imageBox)

        verticalBox1   = BoxLayout(orientation='vertical',size_hint=(0.1,1),spacing=5)

        button5         = Label(text="Select Model",font_size='12sp')

        check_box      = BoxLayout(orientation='vertical')

        check1=ToggleButton(text='Mobilenet',group='model')
        check1.bind(on_press=self.choose_model1)
        
        check2=ToggleButton(text='FRCNN',group='model')
        check2.bind(on_press=self.choose_model2)
        
        check3=ToggleButton(text='SSD',group='model',state='down')
        check3.bind(on_press=self.choose_model3)

        check_box.add_widget(check1)

        check_box.add_widget(check2)

        check_box.add_widget(check3)
        
        threshBox     = BoxLayout(orientation='horizontal')

        threshButton        = Button(text="Set\nThreshold", font_size='10sp')
        threshButton.bind(on_press=self.set_threshold)
        
        self.threshInput    = TextInput(text='0.5')
        
        threshBox.add_widget(threshButton)

        threshBox.add_widget(self.threshInput)

        labelBox      = BoxLayout(orientation='vertical')        

        filterLabel         = Label(text="Label Filter", font_size='10sp')
                
        label1=ToggleButton(text='person')
        label1.bind(on_press=self.choose_label)
        
        label2=ToggleButton(text='car')
        label2.bind(on_press=self.choose_label)
        
        label3=ToggleButton(text='dog')
        label3.bind(on_press=self.choose_label)
        
        label4=ToggleButton(text='bottle')
        label4.bind(on_press=self.choose_label)
        
        label5=ToggleButton(text='chair')
        label5.bind(on_press=self.choose_label)
        
        labelBox.add_widget(filterLabel)
        
        labelBox.add_widget(label1)
        
        labelBox.add_widget(label2)
        
        labelBox.add_widget(label3)
        
        labelBox.add_widget(label4)
        
        labelBox.add_widget(label5)

        button8         = Button(text="Detect",font_size='15sp')
        button8.bind(on_press=self.detect)

        verticalBox1.add_widget(button5)

        verticalBox1.add_widget(check_box)

        verticalBox1.add_widget(threshBox)

        verticalBox1.add_widget(labelBox)

        verticalBox1.add_widget(button8)


        superBox.add_widget(verticalBox)

        superBox.add_widget(verticalBox0)

        superBox.add_widget(verticalBox1)

        return superBox

    def open_folder(self,button1):
        
        self.folder_path=self.file_chooser.selection
        self.files_list=os.listdir(self.folder_path[0])
        self.img_index=0
        self.image_path=self.folder_path[0]+'\\'+self.files_list[self.img_index]
        self.imageBox.source=self.image_path
        self.imageBox.reload()
        print(self.folder_path)

    def display_next_image(self,button2):
        try:
            self.img_index+=1
            self.image_path=self.folder_path[0]+'\\'+self.files_list[self.img_index]
            self.imageBox.source=self.image_path
            self.imageBox.reload()
        except:
            pass

    def display_prev_image(self,button3):
        try:
            self.img_index-=1
            self.image_path=self.folder_path[0]+'\\'+self.files_list[self.img_index]
            self.imageBox.source=self.image_path
            self.imageBox.reload()
        except:
            pass

    def save_annotation(self,button4):
        
        import xml.etree.ElementTree as ET
        # create the file structure
        # img,boxes,scores,classes
        root=''
        try:
            myfile = open("annotations.xml", "r")
            root = myfile.read()
        except:
            pass
        data = ET.Element('data')
        items = ET.SubElement(data, 'items')
        item1 = ET.SubElement(items, 'item')
        item2 = ET.SubElement(items, 'item')
        item3 = ET.SubElement(items, 'item')
        item1.set('name','image')
        item2.set('name','boxes')
        item3.set('name','class')
        item1.text = str(self.img)
        item2.text = str(self.boxes)
        item3.text = str(self.classes)
        # create a new XML file with the results
        newdata = ET.tostring(data, encoding='unicode')
        root+=newdata
        myfile = open("annotations.xml", "w")
        myfile.write(root)
        myfile.close()

    def choose_model1(self,check1):
        self.model_path=desktop_path+'\\Resolute_AI\\ssd_mobilenet_v1_coco_2018_01_28\\frozen_inference_graph.pb'

    def choose_model2(self,check2):
        self.model_path=desktop_path+'\\Resolute_AI\\faster_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'

    def choose_model3(self,check3):
        self.model_path=desktop_path+'\\Resolute_AI\\ssd_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'
    
    def set_threshold(self,threshButton):
        try:
            self.threshold=float(self.threshInput.text)
        except:
            self.threshInput.text='Invalid Entry'
            
    def choose_label(self,lblbutton):
        if lblbutton.state=='down':
            self.filter_classes.append(lblbutton.text)
        elif lblbutton.state=='normal':
            self.filter_classes.remove(lblbutton.text)
        print(self.filter_classes)

    def create_session(self):
        import tensorflow as tf
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            sess = tf.compat.v1.Session(graph=detection_graph)

        return (detection_graph,sess)

    def detect(self,button4):
        
        self.img=cv2.imread(self.image_path)
        img_shape=self.img.shape
        im=cv2.resize(self.img,(300,300))
        im=np.expand_dims(im,axis=0)
        detection_graph,sess=self.create_session()
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (self.boxes, scores, self.classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor:im})

        detected_image=self.visualize(im,self.boxes,scores,self.classes,num,self.threshold)
        detected_image=cv2.resize(detected_image,(img_shape[1],img_shape[0]))
        cv2.imwrite(desktop_path+'obj_detect.jpg',detected_image)
        self.imageBox.source=desktop_path+'obj_detect.jpg'
        self.imageBox.reload()

    
    def visualize(self,img,boxes,scores,classes,num,threshold):

        im1=np.squeeze(img)
        for i in range(int(num[0])):
            if scores[0][i]>threshold:
                if labels[int(classes[0][i])] in self.filter_classes:
                    y0=int(boxes[0][i][0]*im1.shape[0])
                    x0=int(boxes[0][i][1]*im1.shape[1])
                    y1=int(boxes[0][i][2]*im1.shape[0])
                    x1=int(boxes[0][i][3]*im1.shape[1])
                    cv2.rectangle(im1,(x0,y0),(x1,y1),(255,0,255),1)
                    cv2.putText(im1,labels[int(classes[0][i])],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                    #plt.imshow(im1)

        return im1
            
#BoxLayoutDemo().run()


