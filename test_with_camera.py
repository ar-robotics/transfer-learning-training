import cv2
import numpy as np
import tensorflow as tf


labels = []


def make_boxes(num_detections, detection_boxes, detection_classes, detection_scores):
    """Draw the boxes on the frame and label them

    Args:
        num_detections: Number of detections
        detection_boxes: Bounding box coordinates
        detection_classes: Class indices
        detection_scores: Confidence scores

    Returns:
        Frame with bounding boxes and labels"""
    for i in range(num_detections):
        if detection_scores[i] > 0.4:  # Confidence threshold
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (
                xmin * frame.shape[1],
                xmax * frame.shape[1],
                ymin * frame.shape[0],
                ymax * frame.shape[0],
            )
            cv2.rectangle(
                frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2
            )
            # Draw label
            object_name = labels[int(classes[i])]  # Retrieve the class name
            label = "%s" % (object_name)  # Example: 'cat: 72%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            top = max(top, labelSize[1])
            cv2.rectangle(
                frame,
                (int(left), int(top - round(1.5 * labelSize[1]))),
                (int(left + round(1.5 * labelSize[0])), int(top + baseLine)),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (int(left), int(top)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
    return frame


# Loop over frames from the video file stream
def interpret(frame, height, width, interpreter, input_details, output_details):
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
    input_frame = cv2.resize(frame, (width, height))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], input_frame)
    interpreter.invoke()
    num_detections = int(interpreter.get_tensor(output_details[2]["index"])[0])
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    classes = interpreter.get_tensor(output_details[3]["index"])[0]
    return num_detections, scores, boxes, classes


if __name__ == "__main__":
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(
        model_path="tflite_models/people_detection_2.tflite"
    )
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = int(input_details[0]["shape"][1])
    width = int(input_details[0]["shape"][2])

    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        num_detections, scores, boxes, classes = interpret(
            frame, height, width, interpreter, input_details, output_details
        )
        make_boxes(num_detections, boxes, classes, scores)
        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
