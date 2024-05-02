import tensorflow as tf
import os
import cv2
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def load_labels(path):
    """Load labels from text file

    Args:
        path: Path to text file containing labels

    Returns:
        labels: List of labels
    """
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


class Interpreter:
    """Load TFLite model and allocate tensors

    Attributes:
        model_path: Path to TFLite model
        interpreter: TFLite interpreter
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def get_interpreter(self):
        """Get TFLite interpreter

        Returns:
            interpreter: TFLite interpreter
        """
        return self.interpreter

    def get_details(self) -> tuple:
        """Get input and output details

        Returns:
            input_details: Input details
            output_details: Output details
            height: Height of input tensor
            width: Width of input tensor
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = int(input_details[0]["shape"][1])
        width = int(input_details[0]["shape"][2])
        return input_details, output_details, height, width


class Detection:
    """Detect objects in frame using TFLite model

    Attributes:
        frame: Input frame
        height: Frame height
        width: Frame width
        interpreter: TFLite interpreter
        input_detail: Input details
        output_detail: Output details
        scores: Confidence scores
        boxes: Bounding box coordinates
        classes: Class indices
        num_detections: Number of detections
    """

    def __init__(self, frame, height, width, interpreter, input_detail, output_detail):
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

    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression

        Args:
            boxes: Bounding box coordinates
            scores: Confidence scores
            iou_threshold: IoU threshold

        Returns:
            keep: Indices of boxes to keep
        """
        # Convert boxes from [x_center, y_center, width, height]
        # to [x_min, y_min, x_max, y_max]
        x_center, y_center, width, height = (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
        )
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        areas = (x_max - x_min) * (y_max - y_min)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x_min[i], x_min[order[1:]])
            yy1 = np.maximum(y_min[i], y_min[order[1:]])
            xx2 = np.minimum(x_max[i], x_max[order[1:]])
            yy2 = np.minimum(y_max[i], y_max[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

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
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_detail[0]["index"], input_frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_detail[0]["index"]
        )  # get tensor  x(1, 25200, 7)
        xyxy, classes, scores = self.YOLOdetect(output_data)

        boxes_array = np.array(xyxy).T  # Transpose to get shape [num_boxes, 4]

        # Apply NMS
        keep_idxs = self.nms(
            boxes_array, scores, iou_threshold=0.5
        )  # Adjust IoU threshold as needed

        # Filter boxes, scores, and classes based on NMS
        filtered_boxes = boxes_array[keep_idxs]
        filtered_scores = scores[keep_idxs]
        filtered_classes = np.array(classes)[keep_idxs]

        # Update self.boxes, self.scores, self.classes with filtered detections
        self.boxes = filtered_boxes
        self.scores = filtered_scores
        self.classes = filtered_classes
        return self.make_boxes_2()

    def classFilter(self, classdata) -> list:
        """Filter classes

        Args:
            classdata: Class data

        Returns:
            Filtered classes
        """

        classes = []  # create a list
        for i in range(classdata.shape[0]):  # loop through all predictions
            classes.append(
                classdata[i].argmax()
            )  # get the best classification location
        return classes  # return classes (int)

    def YOLOdetect(self, output_data) -> tuple:
        """Detect objects using YOLO

        Args:
            output_data: Output data

        Returns:
            Bounding box coordinates, class indices, and confidence scores
        """
        output_data = output_data[0]  # x(1, 25200, 7) to x(25200, 7)
        boxes = np.squeeze(output_data[..., :4])  # boxes  [25200, 4]
        scores = np.squeeze(output_data[..., 4:5])  # confidences  [25200, 1]
        classes = self.classFilter(output_data[..., 5:])  # get classes
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

        return (
            xyxy,
            classes,
            scores,
        )

    def make_boxes(self, scores, xyxy):
        """Draw bounding boxes on frame

        Args:
            frame: Input frame
            scores: Confidence scores
            xyxy: Bounding box coordinates

        Returns:
            Frame with bounding boxes
        """
        for i in range(len(scores)):
            if (scores[i] > 0.1) and (scores[i] <= 1.0):
                H = self.frame.shape[0]
                W = self.frame.shape[1]
                xmin = int(max(1, (xyxy[0][i] * W)))
                ymin = int(max(1, (xyxy[1][i] * H)))
                xmax = int(min(H, (xyxy[2][i] * W)))
                ymax = int(min(W, (xyxy[3][i] * H)))

                cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

    def make_boxes_2(self):
        """Draw bounding boxes on frame version 2

        Returns:
            Frame with bounding boxes
        """
        num_detections = len(self.scores)
        for i in range(num_detections):
            # Now, we ensure we're only accessing valid indices
            if (self.scores[i] > 0.3) and (self.scores[i] <= 1.0):
                # Assuming self.boxes is now [num_detections, 4] with
                # [xmin, ymin, xmax, ymax] for each detection
                xmin, ymin, xmax, ymax = self.boxes[i]
                xmin = int(max(1, xmin * self.width))
                ymin = int(max(1, ymin * self.height))
                xmax = int(min(self.height, xmax * self.width))
                ymax = int(min(self.width, ymax * self.height))
                cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        return self.frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    labels = load_labels(
        "C:/Users/aditi/Documents/Bachelor_p/Object_Detection-pi/pre-trained model/labels/labels-ppl.txt"  # noqa
    )
    interpreter = Interpreter(
        "C:/Users/aditi/Downloads/all_ml_related/yolov5-20240410T063421Z-001/yolov5/runs/train/exp/weights/best-fp16.tflite"  # noqa
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

        frame = detection_obj.interpret()

        # frame = detection_obj.make_boxes()
        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
