import cv2
import numpy as np
import tensorflow as tf
from pymongo import MongoClient


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mongodbVSCodePlaygroundDB"]

# Initialize video capture
cap = cv2.VideoCapture(0)


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


labels = load_labels("labels-ssd.txt")


class Interpreter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def get_interpreter(self):
        return self.interpreter

    def get_details(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = int(input_details[0]["shape"][1])
        width = int(input_details[0]["shape"][2])
        return input_details, output_details, height, width


class Detection:
    def __init__(
        self, frame, height, width, interpreter, input_detail, output_detail
    ):  # noqa
        self.frame = frame
        self.height = height
        self.width = width
        self.interpreter = interpreter
        self.input_detail = input_detail
        self.output_detail = output_detail
        self.scores = None
        self.boxes = {}
        self.classes = None
        self.num_detections = None

    def interpret(self):
        """Interpret the frame and make predictions

        Args:
            frame: Input frame
            height: Frame height
            width: Frame width
            interpreter: TFLite interpreter
            input_details: Input details
            output_details: Output details

        Returns:
            num_detections: Number of detections
            scores: Confidence scores
            boxes: Bounding box coordinates
            classes: Class indices
        """
        input_frame = cv2.resize(self.frame, (self.width, self.height))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_detail[0]["index"], input_frame)
        self.interpreter.invoke()
        self.num_detections = int(
            self.interpreter.get_tensor(self.output_detail[3]["index"])[0]
        )
        self.scores = self.interpreter.get_tensor(  # noqa
            self.output_detail[2]["index"]
        )[0]
        self.boxes = self.interpreter.get_tensor(  # noqa
            self.output_detail[0]["index"]
        )[0]
        self.classes = self.interpreter.get_tensor(  # noqa
            self.output_detail[1]["index"]
        )[0]

    def make_boxes(self):
        """Draw the boxes on the frame and label them

        Args:
            num_detections: Number of detections
            detection_boxes: Bounding box coordinates
            detection_classes: Class indices
            detection_scores: Confidence scores

        Returns:
            Frame with bounding boxes and labels"""
        for i in range(len(self.scores)):
            if self.scores[i] > 0.4:  # Confidence threshold
                ymin, xmin, ymax, xmax = self.boxes[i]
                (left, right, top, bottom) = (
                    xmin * frame.shape[1],
                    xmax * frame.shape[1],
                    ymin * frame.shape[0],
                    ymax * frame.shape[0],
                )
                cv2.rectangle(
                    frame,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    (0, 255, 0),
                    2,  # noqa
                )
                # Retrieve the class name
                object_name = labels[int(self.classes[i])]
                query = {"type": object_name}
                items_info = employment.find_one(query)
                all_info_dict = {}

                # Loop through all key-value pairs in the items_info document
                for key, value in items_info.items():
                    # Save each key-value pair into the dictionary
                    all_info_dict[key] = value
                print(all_info_dict)
                # Draw label
                # label = "%s" % (object_name)  # Example: 'cat: 72%'
                label_line_1 = f"{object_name}"  # , Age: {age}"
                # label_line_2 = (
                #     f"Date of Employment:
                # {date_of_employment.strftime('%Y-%m-%d')}"
                # )

                (text_width, text_height), baseLine = cv2.getTextSize(
                    label_line_1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                # Set the starting Y position for the text
                text_y_start = int(top - round(1.5 * text_height))

                # Draw the first line of text
                cv2.putText(
                    frame,
                    label_line_1,
                    (int(left), text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                """Adjust Y position for the second line
                based on the height of the first line
                """
                text_y_start += text_height + baseLine

                # Draw the second line of text
                cv2.putText(
                    frame,
                    label_line_2,
                    (int(left), text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        return frame


employment = db["classes"]


interpreter = Interpreter(
    "C:/Users/aditi/Documents/Bachelor_p/Object_Detection-pi/pre-trained model/tflite-models/ssd_mobilenet.tflite"
)
input_details, output_details, height, width = interpreter.get_details()
interpreter = interpreter.get_interpreter()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects
    detection_obj = Detection(
        frame, height, width, interpreter, input_details, output_details
    )
    detection_obj.interpret()
    frame = detection_obj.make_boxes()

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
