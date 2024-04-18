import tensorflow as tf
import numpy as np
import cv2


saved_model_path = "C:/Users/aditi/Documents/Bachelor_p/weights/best_saved_model/"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

with open("people_yolov5_float32.tflite", "wb") as f:
    f.write(tflite_model)
