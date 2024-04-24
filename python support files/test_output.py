import tensorflow as tf
from PIL import Image
import numpy as np

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(
    model_path="tflite_models/ssd_mobilenet.tflite"
)  # noqa
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

i = 0
# print(output_details)
for output_detail in output_details:

    print(i, output_detail)
    i += 1

