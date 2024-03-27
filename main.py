# Webcam Object Detection Using Tensorflow-trained Classifier

# Description:
# This program uses a TensorFlow Lite model to perform
# object detection on a live webcam
# feed. It draws boxes and scores around the objects
# of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a
# separate thread from the main program.


# Import packages
import os
import argparse
import cv2

# import numpy as np
import time
from Database import Database
from Detection import Interpret, Detection

# from Videostream import VideoStream
from threading import Thread


# Define VideoStream class to handle streaming of video in separate thread
# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")  # noqa
        )
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam res in WxH. webcam should support res",
    default="1280x720",
)

args = parser.parse_args()

MODEL_NAME = args.modeldir
print("model name is ", MODEL_NAME)

GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split("x")
imW, imH = int(resW), int(resH)


# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which has model that is used for obj_det
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
print("path to ckpt is ", PATH_TO_CKPT)
print("path to labels is ", PATH_TO_LABELS)

# interpreter = Interpret(PATH_TO_CKPT, PATH_TO_LABELS)
# interpreter = interpreter.get_interpreter()
# input_details, output_details, height, width = interpreter.get_details()
# labels = interpreter.load_labels()

"""if model can be float32
floating_model = input_details[0]["dtype"] == np.float32
input_mean = 127.5
input_std = 127.5
# Normalize pixel values if using a
# floating model (i.e. if model is non-quantized)
# if floating_model:
        print("floating model true")
        input_data = (np.float32(input_data) - input_mean) / input_std
"""


# Initialize frame rate calculation
def frame_rate_calculation():
    """Calculate and return the frame rate"""
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    return frame_rate_calc, freq


videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)
db_name = "mongodbVSCodePlaygroundDB"
collection_name = "classes"
database = Database(db_name, collection_name)
object_name = "person"
query = {"type": object_name}
items_info = database.find_data(query)
print("items are ", type(items_info), items_info)
collection = database.get_collection()
interpret = Interpret(PATH_TO_CKPT, PATH_TO_LABELS)
labels = interpret.load_labels()
print("labels are ", labels)
interpreter = interpret.get_interpreter()
input_details, output_details, height, width = interpret.get_details()
detection_obj = Detection(
    labels,
    height,
    width,
    interpreter,
    input_details,
    output_details,
    collection,  # noqa
)
while True:

    #     # Start timer (for calculating frame rate)
    #     # t1 = cv2.getTickCount()

    #     # Grab frame from video stream
    frame1 = videostream.read()
    #     # Acquire frame and resize to expected shape [1xHxWx3]
    detection_obj.interpret(frame1)
    frame = detection_obj.make_boxes()
    #     # # Draw framerate in corner of frame
    #     # cv2.putText(
    #     #     frame,
    #     #     "FPS: {0:.2f}".format(frame_rate_calc),
    #     #     (30, 50),
    #     #     cv2.FONT_HERSHEY_SIMPLEX,
    #     #     1,
    #     #     (255, 255, 0),
    #     #     2,
    #     #     cv2.LINE_AA,
    #     # )

    #     # All the results have been drawn on the frame,time to display
    cv2.imshow("Object detector", frame)
    #     # t2 = cv2.getTickCount()
    #     # time1 = (t2 - t1) / freq
    #     # frame_rate_calc = 1 / time1
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
videostream.stop()


# if __name__ == "__main__":
#     main()
