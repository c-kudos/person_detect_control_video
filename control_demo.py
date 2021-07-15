import android, time, sys

droid = android.Android()

title = "AidBot Laucher..." 
sstr = "initing and Loading models..."
droid.dialogCreateHorizontalProgress(title,sstr,100) 
droid.dialogShow() 
	
for x in range(0,50):
    time.sleep(0.001) 
    droid.dialogSetCurrentProgress(x) 

from cvs import *

import os
import time
# import cv2
droid.dialogSetCurrentProgress(60) 
import numpy as np
# import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
import tflite_gpu
tflite=tflite_gpu.tflite()

droid.dialogSetCurrentProgress(70) 
# droid.dialogDismiss()

from utils.ssd_mobilenet_utils import *

# from track import KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack
# from tracker import Tracker
from utils.SimpleTracker import SimpleTracker

# from cvp import *

import argparse
def basic_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default='./M_DC.flv', help='camera url or video path')
    parser.add_argument('--interval', type=int, default=0, help='frame interval. equal 1 is for every frame')
    parser.add_argument('--savepath', type=str, default='savepath', help='path for frame images')
    args = parser.parse_args()
    return args


def run_detection(input_shape,image):
    # Run model: start to detect
    # Sets the value of the input tensor.
    tflite.setTensor_Int8(image,input_shape[1],input_shape[1])
    # Invoke the interpreter.
    tflite.invoke()
    
    # print(input_details.shape,output_details.shape)

    # get results
    boxes = tflite.getTensor_Fp32(0)
    classes = tflite.getTensor_Fp32(1)
    scores = tflite.getTensor_Fp32(2)
    num = tflite.getTensor_Fp32(3)

    box=boxes.reshape((10,4))
    # print("boxes",box)
    # print("scores",scores)
    # print("classes",classes)
    # print("num",num)
    
    box, scores, classes = np.squeeze(box), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
    
    # print("boxes1",boxes)
    # print("scores1",scores)
    # print("classes1",classes)
    # print("num1",num)
    
    out_scores, out_boxes, out_classes = non_max_suppression(scores, box, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))
            
    return out_scores, out_boxes, out_classes

def image_object_detection(interpreter, colors):
    image = cv2.imread('images/dog.jpg')
    image_data = preprocess_image_for_tflite(image, model_image_size=300)
    out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)

    # Draw bounding boxes on the image file
    result = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    cv2.imwrite(os.path.join("out", "ssdlite_mobilenet_v2_dog.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_object_detection(droid,speed,input_shape, colors,class_names):
    camera = cvs.VideoCapture(0)
    parser = basic_options()
    cap = cv2.VideoCapture(parser.cam)
    # camera=cvp.VideoCapture(4,0)
    
    if not os.path.exists(parser.savepath):
        os.makedirs(parser.savepath)
        
    framenum = 0
    while True:
        start = time.time()
        frame = cvs.read() 

        if frame is not None:
            # image_data = preprocess_image_for_tflite(frame, model_image_size=300)
            image_data = preprocess_image_for_tflite_uint8(frame, model_image_size=300)

            # print(image_data)
            out_scores, out_boxes, out_classes = run_detection(input_shape,image_data)
            # print("out_boxes",out_boxes)
            # print("out_scores",out_scores)
            # print("out_classes",out_classes)
            
            end = time.time()
  
            if (out_classes == 1).any():
                ret, frame = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS)
                framenum += 1
                cv2.imwrite(os.path.join(parser.savepath, 'pause.jpg'), frame)
                cvs.imshow(frame)
            else :
                image = cvs.imread("./savepath/pause.jpg")
                cvs.imshow(image)
            
          
    # camera.release()
    # cv2.destroyAllWindows()

def main():
    # Load TFLite model and allocate tensors.
    # import tflite_runtime.interpreter as tflite
    
    inShape =[1 * 300 * 300 *3,]
    outShape= [1 * 10*4*4,1*10*4,1*10*4,1*4]
    model_path="model_data/ssdlite_mobilenet_v3.tflite"
    print('gpu:',tflite.NNModel(model_path,inShape,outShape,4,0))#4表示4个线程，0表示gpu，-1表示cpu，1表示NNAPI
    
    # interpreter = Interpreter(model_path="model_data/ssdlite_mobilenet_v3.tflite",num_threads=4)
    # interpreter.allocate_tensors()

    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    input_shape=[300,300]


    # print("=======================================")
    # print("input :", str(input_details))

    # print("ouput :", str(output_details))
    # print("=======================================")

    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
            
    #image_object_detection(interpreter, colors)
    
    print("connecting to usb bot")
    r=droid.connectUsb()
    print(r)
    print("sendIndicatorToVehicle: 0")
    droid.sendIndicatorToVehicle(0)
    speed=92;#range(1,250)
    real_time_object_detection(droid,speed,input_shape, colors,class_names)



class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)
    
    def idle(self):
        #idle function called every update cycle
        self.lbl.set_text(cvs.getLbs())
        self.aidcam.update()
        pass
        
    def main(self):
        #creating a container VBox type, vertical (you can use also HBox or Widget)
        main_container = VBox(width=360, height=520, style={'margin':'0px auto'})
        main_container.css_width = "98%"
        # main_container.css_height = "80%"
        
        self.aidcam = OpencvVideoWidget(self, width=360, height=320)
        self.aidcam.css_width = "98%"
        # self.aidcam.css_height = "80%"
        # self.aidcam.style['margin'] = '10px'
        
        self.aidcam.identifier="myimage_receiver"
        main_container.append(self.aidcam)
        
        self.lbl = Label('This show fps!', width=360, height=30,  margin='50px',)
        main_container.append(self.lbl)

        return main_container
    
    

droid.dialogSetCurrentProgress(100) 
droid.dialogDismiss()

main()
