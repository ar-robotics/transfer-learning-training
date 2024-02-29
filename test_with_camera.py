import cv2
import numpy as np
import tensorflow as tf

#Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="tflite_models/people_detection_2.tflite")
interpreter.allocate_tensors()

#Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    print(labels)

#Initialize video capture
cap = cv2.VideoCapture(0)



def make_boxes(num_detections, detection_boxes, detection_classes, detection_scores):
    
    """ Draw the boxes on the frame and label them
    Args:
    num_detections: Number of detections
    detection_boxes: Bounding box coordinates
    detection_classes: Class indices
    detection_scores: Confidence scores
    Returns:
        None """
    for i in range(num_detections):
        if detection_scores[i] > 0.4:  # Confidence threshold
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                        ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
             #Draw label
            object_name = labels[int(classes[i])]  # Retrieve the class name
            label = '%s' % (object_name)  # Example: 'cat: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*labelSize[1]))), (int(left + round(1.5*labelSize[0])), int(top + baseLine)), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame

            
           

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Resize frame to model input dimensions
    input_frame = cv2.resize(frame, (width, height))
    
    #Convert frame to float32 and normalize (if your model requires normalization)
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
    #input_frame /= 255.0  # Uncomment this if your model expects input values to be normalized
    
    #Set the model input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    
    #Retrieve detection results
    num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
    
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    #print(scores.shape)# Confidence scores
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
    #print(boxes.shape)# Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class inde
    #print(classes.shape)
    #Draw boxes
    make_boxes(num_detections, boxes, classes, scores)
    #Display the resulting frame
    cv2.imshow('Object Detection', frame)
    
    #Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release capture
cap.release()
cv2.destroyAllWindows()
