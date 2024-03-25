from Database import Database
from Detection import Interpreter, Detection
import cv2

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


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects
    detection_obj = Detection(
        frame,
        labels,
        height,
        width,
        interpreter,
        input_details,
        output_details,
        collection,  # noqa
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
