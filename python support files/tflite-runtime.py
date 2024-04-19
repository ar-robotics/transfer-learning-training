# Imports
import tflite_support
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

model_path = "tflite_models/best-int8.tflite"
# Initialization
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(max_results=1)
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options
)
# detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
detector = vision.ObjectDetector.create_from_file(model_path)
image_path = "iii.jpg"
# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)
