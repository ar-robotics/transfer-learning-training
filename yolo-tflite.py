import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf


# Initialize video capture
cap = cv2.VideoCapture(0)


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


labels = load_labels(
    "C:/Users/aditi/Documents/Bachelor_p/Object_Detection-pi/pre-trained model/labels/labels-ppl.txt"
)


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
    def __init__(self, height, width, interpreter, input_detail, output_detail):  # noqa
        self.h = height
        self.w = width
        self.interpreter = interpreter
        self.input_details = input_detail
        self.output_details = output_detail

    def nms(self, boxes, scores, iou_threshold):

        return

    def interpret(self, frame):
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
        input_frame = cv2.resize(frame, (self.w, self.h))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_frame)  # noqa
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )  # get tensor  x(1, 25200, 7)
        xyxy, classes, scores = self.YOLOdetect(output_data)
        frame = self.make_boxes_2(frame, xyxy, classes, scores)
        return frame

    def classFilter(self, classdata):

        classes = []  # create a list
        for i in range(classdata.shape[0]):  # loop through all predictions
            classes.append(
                classdata[i].argmax()
            )  # get the best classification location
        return classes  # return classes (int)

    def YOLOdetect(self, output_data):
        output_data = output_data[0]  # x(1, 25200, 7) to x(25200, 7)
        boxes_tensor = output_data[..., :4]  # boxes  [25200, 4]
        scores_tensor = output_data[..., 4:5]  # confidences  [25200, 1]
        classes_tensor = output_data[..., 5:]  # classes  [25200, 3]

        boxes = tf.reshape(
            boxes_tensor, [1, -1, 1, 4]
        )  # Reshape to [batch_size, num_boxes, num_classes, 4]

        scores = tf.reshape(
            scores_tensor, [1, -1, 1]
        )  # Reshape to [batch_size, num_boxes, num_classes]

        # Apply combined_non_max_suppression
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
            tf.image.combined_non_max_suppression(
                boxes=boxes,
                scores=scores,
                max_output_size_per_class=10,
                max_total_size=10,
                iou_threshold=0.5,
                score_threshold=0.5,
            )
        )
        boxes = nmsed_boxes.numpy()
        scores = nmsed_scores.numpy()

        classes = self.classFilter(output_data[..., 5:])  # get classes
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

        return (
            xyxy,
            classes,
            scores,
        )

    def make_boxes_2(self, frame, xyxy, classes, scores):
        for i in range(len(scores)):
            if (scores[i] > 0.9) and (scores[i] <= 1.0):
                print(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                xmin = int(max(1, (xyxy[0][i] * self.w)))
                ymin = int(max(1, (xyxy[1][i] * self.h)))
                xmax = int(min(self.h, (xyxy[2][i] * self.w)))
                ymax = int(min(self.w, (xyxy[3][i] * self.h)))
                cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2  # noqa
                )
                cv2.putText(
                    frame,
                    "{}: {:.2f}".format(labels[classes[i]], scores[i]),
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (10, 255, 0),
                    2,
                )
        return frame


interpreter = Interpreter(
    "C:/Users/aditi/Downloads/all_ml_related/yolov5-20240410T063421Z-001/yolov5/runs/train/exp/weights/best-fp16.tflite "  # noqa
)
input_details, output_details, height, width = interpreter.get_details()
interpreter = interpreter.get_interpreter()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects
    detection_obj = Detection(
        height, width, interpreter, input_details, output_details
    )  # noqa

    frame = detection_obj.interpret(frame)

    # frame = detection_obj.make_boxes()
    # Display the resulting frame
    # cv2.imshow("Object Detection", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
