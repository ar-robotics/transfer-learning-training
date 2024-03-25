import cv2
import numpy as np
import tensorflow as tf


class Interpreter:
    def __init__(self, model_path, label_path):
        """Initialize the TFLite interpreter

        Args:
            model_path: Path to the TFLite model
            label_path: Path to the label file
        """
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.label_path = label_path
        self.interpreter.allocate_tensors()

    def load_labels(self):
        """Load the labels from the label file

        Returns:
            labels: List of labels
        """
        with open(self.label_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_interpreter(self):
        """Get the TFLite interpreter

        Returns:
            interpreter: TFLite interpreter
        """
        return self.interpreter

    def get_details(self):
        """Get the input and output details of the TFLite model

        Returns:
            input_details: Input details
            output_details: Output details
            height: Height of the input
            width: Width of the input
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = int(input_details[0]["shape"][1])
        width = int(input_details[0]["shape"][2])
        return input_details, output_details, height, width


class Detection:
    def __init__(
        self,
        frame,
        labels,
        height,
        width,
        interpreter,  # noqa
        input_detail,
        output_detail,
        collection,
    ):
        self.frame = frame
        self.height = height
        self.width = width
        self.labels = labels
        self.interpreter = interpreter
        self.input_detail = input_detail
        self.output_detail = output_detail
        self.scores = None
        self.boxes = {}
        self.classes = None
        self.num_detections = None
        self.collection = collection

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

    def retreive_info(self, object_name):
        """
        Retrieve all information for the object from the database

        Args:
            database: Database object
            object_name: Name of the object

        Returns:
            all_info_dict: Dictionary containing all information for the object
        """
        query = {"type": object_name}
        items_info = self.collection.find_one(query)
        all_info_dict = {}
        for key, value in items_info.items():
            # Save each key-value pair into the dictionary
            if key != "_id" and key != "type":
                all_info_dict[key] = value
        return all_info_dict

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
            if self.scores[i] > 0.8:  # Confidence threshold
                ymin, xmin, ymax, xmax = self.boxes[i]
                (left, right, top, bottom) = (
                    xmin * self.frame.shape[1],
                    xmax * self.frame.shape[1],
                    ymin * self.frame.shape[0],
                    ymax * self.frame.shape[0],
                )
                cv2.rectangle(
                    self.frame,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    (0, 255, 0),
                    2,  # noqa
                )
                object_name = self.labels[int(self.classes[i])]
                # dict_info = self.retreive_info(object_name)
                # Loop through all key-value pairs in the items_info document
                label_with_score = "{}: {:.2f}".format(
                    object_name, self.scores[i]  # noqa
                )
                # y_offset = ymin - 25
                # (text_width, text_height), baseLine = cv2.getTextSize(
                #     label_line_1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                # )
                # cv2.rectangle(
                #     self.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2
                # )  # noqa
                cv2.putText(
                    self.frame,
                    label_with_score,
                    (int(xmin), int(ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                # for key, value in dict_info.items():
                #     # Generate the text for this key-value pair
                #     info_text = "{}: {}".format(key, value)
                #     # Display this key-value pair on the frame
                #     cv2.putText(
                #         self.frame,
                #         info_text,
                #         (int(xmin), int(y_offset)),
                #         cv2.FONT_ITALIC,
                #         0.5,  # Yo
                #         (
                #             255,
                #             0,
                #             0,
                #         ),
                #         2,
                #     )
                #     y_offset -= 20
        return self.frame
