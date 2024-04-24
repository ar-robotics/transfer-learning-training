# import tensorflow as tf
# import numpy as np
import cv2


# saved_model_path = "C:/Users/aditi/Documents/Bachelor_p/weights/best_saved_model/"
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
# tflite_model = converter.convert()

# with open("people_yolov5_float32.tflite", "wb") as f:
#     f.write(tflite_model)


# Assuming 'frame' is your loaded or captured frame
frame = cv2.imread('t.jpg') 
(h,w) = frame.shape[:2]
end_effector_point = (w//2, h//2)  # Example point
# Draw the circle
cv2.circle(frame, end_effector_point, radius=2, color=(0, 0, 255), thickness=-1)

# Display the image
cv2.imshow('Frame with Circle', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
