######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Tushar Bhanarkar
# Date: 5/30/21
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam feed
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py


#!/usr/bin/env python3.8

import cv2
import numpy as np
import time

import tflite_runtime.interpreter as tflite
from PIL import Image

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

top_k_results = 3

model_path = '/root/new_tf/mobilenet_v1_1.0_224_quant.tflite'
label_path = '/root/new_tf/labels.txt'

def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

print("Loading module")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
    

print("Module loaded... running interpreter")

if __name__ == "__main__":
    
    print("...Camera Execution...")    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 10)

   
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #get width and height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    
    floating_model = input_details[0]['dtype'] == np.float32
    #process Stream

    while True:
        ret, frame = cap.read()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        input_data = np.expand_dims(image, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        results = np.squeeze(predictions)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_path)
        for i in top_k:
            if floating_model:
              print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
              print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
                       
       

            #break

    cap.release()
    cv2.destroyAllWindows()
