# import tensorflow as tf
# from PIL import Image
# import numpy as np

# # Load TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(
#     model_path="tflite_models/ssd_mobilenet.tflite"
# )  # noqa
# interpreter.allocate_tensors()

# # Get model details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# i = 0
# # print(output_details)
# for output_detail in output_details:

#     print(i, output_detail)
#     i += 1

def check_file_encoding(file_path):
    try:
        with open(file_path, encoding='utf-8') as file:
            file.read()
        print("No encoding issues detected.")
    except UnicodeDecodeError as e:
        print(f"Encoding issue detected: {e}")

# Replace 'your_notebook.ipynb' with the path to your notebook
# check_file_encoding('docs/notebooks/Custom_training_bolt_detection.ipynb')
