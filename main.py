from Database import Database
from Detection import Interpreter, Detection
import cv2
from threading import Thread

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize database
db_name = "mongodbVSCodePlaygroundDB"
collection_name = "classes"
database = Database(db_name, collection_name)
collection = database.get_collection()

interpreter = Interpreter(
    "Object_Detection-pi/pre-trained model/tflite-models/ssd_mobilenet.tflite",
    "transfer-learning-training/labels-ssd.txt",
)
labels = interpreter.load_labels()
input_details, output_details, height, width = interpreter.get_details()
interpreter = interpreter.get_interpreter()


def get_frame() -> tuple:
    ret, frame = cap.read()
    return ret, frame


# Detect objects
detection_obj = Detection(
    labels,
    height,
    width,
    interpreter,
    input_details,
    output_details,
    collection,  # noqa
)
thread = Thread(target=detection_obj.analyze, daemon=True)
thread.start()

while True:
    ret, frame = get_frame()
    detection_obj.set_frame(ret, frame)
    # detection_obj.analyze()
    # Display the resulting frame
    if (
        detection_obj.display_frame is not None
    ):  # Ensure there is a display_frame to show
        cv2.imshow("Object Detection", detection_obj.display_frame)
    else:
        cv2.imshow("Object Detection", frame)
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
