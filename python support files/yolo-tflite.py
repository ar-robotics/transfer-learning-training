import numpy as np
import tensorflow as tf
import cv2

cap = cv2.VideoCapture(0)


def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):  # loop through all predictions
        classes.append(classdata[i].argmax())  # get the best classification location
    return classes  # return classes (int)


def YOLOdetect(
    output_data,
):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]  # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])  # boxes  [25200, 4]

    scores = np.squeeze(output_data[..., 4])  # confidences  [25200, 1]

    classes = classFilter(output_data[..., 5])
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    # print(xyxy)
    indices = nms(xyxy, scores, 0.5, 0.5)  # perform non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        class_id = classes[i]

        # Draw the detection on the input image
        return (box, class_id, scores)


def nms(boxes, scores, confidence_thres, iou_threshold):
    boxes = np.array(boxes).transpose()
    boxes_list = boxes.tolist()
    indices = cv2.dnn.NMSBoxes(boxes_list, scores, confidence_thres, iou_threshold)
    print(indices)
    return indices


def make_boxes(frame, scores, classes, output_data, labels=["person"]):
    outputs = np.transpose(np.squeeze(output_data[0]))
    rows = outputs.shape[0]
    print(scores)
    for i in range(rows):
        if (scores[i] > 0.7) and (scores[i] <= 1.0):
            H = frame.shape[0]
            W = frame.shape[1]
            xmin = int(max(1, (xyxy[0][i] * W)))
            ymin = int(max(1, (xyxy[1][i] * H)))
            xmax = int(min(H, (xyxy[2][i] * W)))
            ymax = int(min(W, (xyxy[3][i] * H)))
            labelname = labels[classes[i]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(
                frame,
                labelname + ":" + str(scores[i]),
                (xmin, ymin - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3 * H,
                (255, 0, 0),
                1,
            )


interpreter = tf.lite.Interpreter(
    model_path="C:/Users/aditi/Documents/Bachelor_p/people_yolov5_float32.tflite"
)  # noqa
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
while True:
    ret, frame = cap.read()
    image_data = cv2.resize(frame, (640, 640))
    image_data = np.array(image_data).astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)
    interpreter.set_tensor(input_details[0]["index"], image_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    xyxy, classes, scores = YOLOdetect(output_data)
    make_boxes(frame, scores, classes, output_data)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break