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


class Process:
    def __init__(
        self, frame, height, width, interpreter, input_details, output_details
    ):
        self.frame = frame
        self.height = height
        self.width = width
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def draw_detections(self, frame, box, score, class_id):
        left, top, right, bottom = box
        cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)
        cv2.putText(
            frame,
            "{}: {:.2f}".format("Person", score),
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 255, 0),
            2,
        )

    def preprocess(self):
        frame = cv2.resize(self.frame, (self.width, self.height))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def postprocess(self, frame, output):
        output = np.transpose(np.squeeze(output[0]))
        rows = output.shape[0]
        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):

            classes_scores = output[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= 0.9:
                class_id = np.argmax(classes_scores)

                x, y, w, h = output[i][0], output[i][1], output[i][2], output[i][3]
                left = int((x - w / 2) * self.width)
                top = int((y - h / 2) * self.height)
                right = int((x + w / 2) * self.width)
                bottom = int((y + h / 2) * self.height)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, right, bottom])

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.9, 0.4)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            self.draw_detections(frame, box, score, class_id)
        return frame

    def main(self):

        frame = self.preprocess()
        self.interpreter.set_tensor(self.input_details[0]["index"], frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        frame = self.postprocess(self.frame, output_data)
        return frame


class Detection:
    def __init__(self, height, width, interpreter, input_detail, output_detail):  # noqa
        self.h = height
        self.w = width
        self.interpreter = interpreter
        self.input_details = input_detail
        self.output_details = output_detail

    def interpret(self, frame):
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
        classes = output_data[..., 5:]  # classes  [25200, 3]

        boxes = np.squeeze(boxes_tensor)  # [25200, 4]
        scores = np.squeeze(scores_tensor)  # [25200, 1]
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
    # detection_obj = Detection(
    #     height, width, interpreter, input_details, output_details
    # )  # noqa
    # frame = detection_obj.interpret(frame)
    detection_obj = Process(
        frame, height, width, interpreter, input_details, output_details
    )
    frame = detection_obj.main()
    cv2.imshow("Object Detection", frame)
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture
