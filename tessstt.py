# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="model_fin_3.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Read the image and preprocess it
# image = Image.open('OIDv4_ToolKit/OID/train_All/4a78838bf777e9c1.jpg')

# image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
# image = np.expand_dims(image, axis=0)
# image = image.astype(np.uint8)

# # Set the tensor to point to the input data to be inferred
# interpreter.set_tensor(input_details[0]['index'], image)

# # Run the inference
# interpreter.invoke()

# # Extract the output data
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

# predicted_class_index = np.argmax(output_data)
# predicted_class_score = np.max(output_data)
# print(f"Predicted class index: {predicted_class_index}, Score: {predicted_class_score}")

# with open('labels.txt', 'r') as file:
#     labels = [line.strip() for line in file.readlines()]
# predicted_class_name = labels[predicted_class_index]
# print(f"Predicted class name: {predicted_class_name}")
# import cv2
# import numpy as np
# import tensorflow as tf

# # Load TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path="model_fin_3.tflite")
# interpreter.allocate_tensors()

# # Get model details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# #print(output_details)
# height = input_details[0]['shape'][1]
# width = input_details[0]['shape'][2]

# # Load labels
# with open('labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Resize frame to model input dimensions
#     input_frame = cv2.resize(frame, (width, height))
    
#     # Convert frame to float32 and normalize (if your model requires normalization)
#     input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
#     # input_frame /= 255.0  # Uncomment this if your model expects input values to be normalized
    
#     # Set the model input and run inference
#     interpreter.set_tensor(input_details[0]['index'], input_frame)
#     interpreter.invoke()
    
#     # Retrieve detection results
#     num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
    
#     scores = interpreter.get_tensor(output_details[0]['index'])[0]
#     #print(scores.shape)# Confidence scores
#     boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
#     #print(boxes.shape)# Bounding box coordinates
#     classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class inde
#     #print(classes.shape)
#     # Iterate over detections and draw bounding boxes on the original frame
#     for i in range(num_detections):
#         if scores[i] > 0.2:  # Confidence threshold
#             ymin, xmin, ymax, xmax = boxes[i]
#             (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
#                                         ymin * frame.shape[0], ymax * frame.shape[0])
#             cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
#             # Draw label
#             object_name = labels[int(classes[i])]  # Retrieve the class name
#             label = '%s: %d%%' % (object_name, int(scores[i]*100))  # Example: 'cat: 72%'
#             labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
#             top = max(top, labelSize[1])
#             cv2.rectangle(frame, (int(left), int(top - round(1.5*labelSize[1]))), (int(left + round(1.5*labelSize[0])), int(top + baseLine)), (255, 255, 255), cv2.FILLED)
#             cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Object Detection', frame)
    
#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release capture
# cap.release()
# cv2.destroyAllWindows()


# import numpy as np
# import tensorflow as tf
# from PIL import Image, ImageDraw

# # Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path="model_fin_2.tflite")
# interpreter.allocate_tensors()

# # Get model input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Function to load an image and resize it to fit the model input
# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((320, 320),Image.LANCZOS)
#     return np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

# # Function to run inference on an image and parse the output
# def run_inference(image_path):
#     # Load and process the image
#     input_image = load_image(image_path)
    
#     # Set the tensor to point to the input data to be inferred
#     interpreter.set_tensor(input_details[0]['index'], input_image)
    
#     # Run the inference
#     interpreter.invoke()
    
#     # Retrieve detection results
#     scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores
#     boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding boxes
#     num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])  # Number of detections
#     classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class IDs
    
#     return scores, boxes, num_detections, classes

# # Function to display detected objects on the image
# def display_detections(image_path, scores, boxes, num_detections, classes):
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
    
#     for i in range(num_detections):
#         if scores[i] > 0.1:  # Threshold can be adjusted
#             ymin, xmin, ymax, xmax = boxes[i]
#             (left, right, top, bottom) = (xmin * image.width, xmax * image.width, 
#                                           ymin * image.height, ymax * image.height)
#             draw.rectangle([left, top, right, bottom], outline="red")
#             # Adjust this to use your class names
#             class_name = f"Class {int(classes[i])}"
#             draw.text((left, top), class_name + " " + str(scores[i]))
    
#     image.show()

# # Example usage
# image_path = "OIDv4_ToolKit/OID/train_All/4a78838bf777e9c1.jpg"
# scores, boxes, num_detections, classes = run_inference(image_path)
# display_detections(image_path, scores, boxes, num_detections, classes)

import cv2
import tensorflow as tf
import numpy as np
interpreter = tf.lite.Interpreter(model_path="model_fin_3.tflite")
interpreter.allocate_tensors()

#input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
    
image = cv2.imread("OIDv4_ToolKit/OID/train_All/4a78838bf777e9c1.jpg")
input_details = interpreter.get_input_details()[0]
input_batch, input_height, input_width, input_channels = input_details['shape']

preprocessed_image = cv2.resize(image, (input_height, input_width))
preprocessed_image = np.expand_dims(preprocessed_image, axis=0).astype(np.uint8)  # Normalize
interpreter.set_tensor(input_details['index'], preprocessed_image)
interpreter.invoke()

output_details = interpreter.get_output_details()
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]
threshold = 0.5  # Adjust as needed

for i in range(len(scores[0])):
    if scores[0][i] >= threshold:
        # Process detection results (e.g., draw bounding boxes on the image)
        ymin, xmin, ymax, xmax = boxes[i]

        # Convert relative coordinates to absolute image coordinates
        image_height, image_width, _ = image.shape
        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)

        # Get the class ID and label
        class_id = int(classes[i])
        label = classes[class_id]

        # Draw the bounding box and label on the image
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(image, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the final image with bounding boxes
cv2.imshow("Detection Results", image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
