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

# image_data = Image.open("sha.png").resize((640, 640))
# image_data = np.array(image_data).astype(np.float32)
# image_data = np.expand_dims(image_data, axis=0)
# interpreter.set_tensor(input_details[0]["index"], image_data)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]["index"])
# output_data = output_data[..., [1, 0, 3, 2]]
# output_data[..., 0:4] = output_data[..., 0:4] * 640
# output_data[..., 5] = np.argmax(output_data[..., 5], axis=-1)

# # Round the confidence score to 2 decimal places
# output_data[..., 4] = np.round(output_data[..., 4], decimals=2)

# # Print the output shape and data
# print(output_data)
