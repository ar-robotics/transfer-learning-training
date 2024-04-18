# import argparse

# import cv2
# import numpy as np
# from tensorflow.lite import interpreter as tflite

# from ultralytics.utils import ASSETS, yaml_load
# from ultralytics.utils.checks import check_yaml


# class Yolov8TFLite:

#     def __init__(self, tflite_model, input_image, confidence_thres, iou_thres):
#         """
#         Initializes an instance of the Yolov8TFLite class.

#         Args:
#             tflite_model: Path to the TFLite model.
#             input_image: Path to the input image.
#             confidence_thres: Confidence threshold for filtering detections.
#             iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
#         """
#         self.tflite_model = tflite_model
#         self.input_image = input_image
#         self.confidence_thres = confidence_thres
#         self.iou_thres = iou_thres

#         # Load the class names from the COCO dataset
#         self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]

#         # Generate a color palette for the classes
#         self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

#     def draw_detections(self, img, box, score, class_id):
#         """
#         Draws bounding boxes and labels on the input image based on the detected objects.

#         Args:
#             img: The input image to draw detections on.
#             box: Detected bounding box.
#             score: Corresponding detection score.
#             class_id: Class ID for the detected object.

#         Returns:
#             None
#         """

#         # Extract the coordinates of the bounding box
#         x1, y1, w, h = box

#         # Retrieve the color for the class ID
#         color = self.color_palette[class_id]

#         # Draw the bounding box on the image
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

#         # Create the label text with class name and score
#         label = f"{self.classes[class_id]}: {score:.2f}"

#         # Calculate the dimensions of the label text
#         (label_width, label_height), _ = cv2.getTextSize(
#             label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
#         )

#         # Calculate the position of the label text
#         label_x = x1
#         label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

#         # Draw a filled rectangle as the background for the label text
#         cv2.rectangle(
#             img,
#             (label_x, label_y - label_height),
#             (label_x + label_width, label_y + label_height),
#             color,
#             cv2.FILLED,
#         )

#         # Draw the label text on the image
#         cv2.putText(
#             img,
#             label,
#             (label_x, label_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 0),
#             1,
#             cv2.LINE_AA,
#         )

#     def preprocess(self):
#         """
#         Preprocesses the input image before performing inference.

#         Returns:
#             image_data: Preprocessed image data ready for inference.
#         """
#         # Read the input image using OpenCV
#         self.img = cv2.imread(self.input_image)

#         # Get the height and width of the input image
#         self.img_height, self.img_width = self.img.shape[:2]

#         # Convert the image color space from BGR to RGB
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

#         # Resize the image to match the input shape (320x320 for this case)
#         img = cv2.resize(img, (self.input_width, self.input_height))

#         # Normalize the image data by dividing it by 255.0
#         image_data = np.array(img) / 255.0

#         # Transpose the image to have the channel dimension as the first dimension
#         # image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

#         # Quantize the image data to int8
#         image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

#         # Return the preprocessed image data
#         return image_data

#     def postprocess(self, input_image, output):

#         return input_image

#     def main(self):
#         """
#         Performs inference using a TFLite model and returns the output image with drawn detections.

#         Returns:
#             output_img: The output image with drawn detections.
#         """
#         # Create an interpreter for the TFLite model
#         interpreter = tflite.Interpreter(model_path=self.tflite_model)
#         interpreter.allocate_tensors()

#         # Get the model inputs
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()

#         # Store the shape of the input for later use
#         input_shape = input_details[0]["shape"]
#         self.input_width = input_shape[1]
#         self.input_height = input_shape[2]

#         # Preprocess the image data
#         img_data = self.preprocess()

#         # Set the input tensor to the interpreter
#         interpreter.set_tensor(input_details[0]["index"], img_data)

#         # Run inference
#         interpreter.invoke()

#         # Get the output tensor from the interpreter
#         output = interpreter.get_tensor(output_details[0]["index"])

#         # Perform post-processing on the outputs to obtain output image.
#         return self.postprocess(self.img, output)  # output image


# if __name__ == "__main__":
#     # Create an argument parser to handle command-line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="yolov8n_float32.tflite",
#         help="Input your TFLite model.",
#     )
#     parser.add_argument(
#         "--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image."
#     )
#     parser.add_argument(
#         "--conf-thres", type=float, default=0.5, help="Confidence threshold"
#     )
#     parser.add_argument(
#         "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
#     )
#     args = parser.parse_args()

#     # Create an instance of the Yolov8TFLite class with the specified arguments
#     detection = Yolov8TFLite(args.model, args.img, args.conf_thres, args.iou_thres)

#     # Perform object detection and obtain the output image
#     output_image = detection.main()

#     # Display the output image in a window
#     cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#     cv2.imshow("Output", output_image)

#     # Wait for a key press to exit
#     cv2.waitKey(0)


import argparse
import cv2
import numpy as np
import tensorflow as tf

# from ultralytics.utils import yaml_load, check_yaml


class Yolov8TFLite:
    def __init__(self, tflite_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8TFLite class to use with video streams.
        """
        self.tflite_model = tflite_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]
        self.classes = {"0": "person"}
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Setup input size
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]

    def process_frame(self, frame):
        """
        Process a single video frame for object detection.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], img_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        self.postprocess(frame, output_data)
        return frame

    def postprocess(self, frame, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        self.img_width, self.img_height = frame.shape[:2]
        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(frame, box, score, class_id)

        # Return the modified input image

    def run(self):
        """
        Starts the video stream and object detection processing.
        """
        cap = cv2.VideoCapture(0)  # Default camera
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame)
                cv2.imshow("Output", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="yolov8n_float32.tflite", help="TFLite model path."
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="Confidence threshold."
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="IoU threshold for NMS."
    )
    args = parser.parse_args()
    model_path = "C:/Users/aditi/Documents/Bachelor_p/weights/best-fp16.tflite"
    detector = Yolov8TFLite(model_path, args.conf_thres, args.iou_thres)
    detector.run()
