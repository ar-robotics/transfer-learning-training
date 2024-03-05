import numpy as np
import os
import cv2
import numpy as np
import tensorflow as tf
import contextlib
from PIL import Image
from matplotlib import pyplot as plt
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from absl import logging

COLORS = None


def load_data(csv_file_path, images_dir):
    """Loads data from the csv file.

    Args:
        csv_file_path: The file path to the csv file.
        images_dir: The directory that contains images.

    Returns:
        A Dataloader containing the data of training, validation and test set.
    """
    train_data, test_data, validation_data = object_detector.DataLoader.from_csv(
        csv_file_path, images_dir=images_dir
    )
    return train_data, test_data, validation_data


def train_model(train_data, validation_data):
    """Fine tunes the EfficientDet-Lite0 model on the given dataset.

    Args:
        train_data: Training dataset, in the form of Dataloader.
         validation_data: Validation dataset, in the form of Dataloader.

    Returns:
         A trained model.
    """
    spec = model_spec.get("efficientdet_lite0")
    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=4,
        train_whole_model=False,
        epochs=25,
        validation_data=validation_data,
    )
    return model


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model

    Args:
        image_path: path to the input image
        input_size: the input size of the model
    Returns:
        preprocessed image in the format of (1, input_size, input_size, 3)
        original image size
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info.

    Args:
        interpreter: tflite.Interpreter
        image: A [1, height, width, 3] Tensor of type tf.uint8.threshold: a
        floating point number

    Returns:
        a list of dicts
    """
    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output["output_0"]))
    scores = np.squeeze(output["output_1"])
    classes = np.squeeze(output["output_2"])
    boxes = np.squeeze(output["output_3"])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results

    Args:
        image_path: path to the input image
        interpreter: tflite.Interpreter
        threshold: a floating point number
    """
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path, (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj["class_id"])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj["score"] * 100)
        cv2.putText(
            original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == "__main__":
    # shows only errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    """check tensorflow version"""
    assert tf.__version__.startswith("2")

    tf.get_logger().setLevel("ERROR")

    logging.set_verbosity(logging.ERROR)
    csv_file_path = "csv_files/people_updated_labels.csv"
    images_dir = "images_zipped/"

    train_data, test_data, validation_data = load_data(csv_file_path, images_dir)

    model = train_model(train_data, validation_data)
    print(model.evaluate(test_data))

    # Redirects the stdout to a file
    with open("out.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            # This will print the evaluation results to 'out.txt' instead of the standard output
            print(model.evaluate(test_data))

    model.export(export_dir=".")  # Exports to the export directory
    model.export(
        export_dir=".", export_format=[ExportFormat.TFLITE, ExportFormat.LABEL]
    )
    model.export(
        export_dir=".", tflite_filename="People_Detection_2.tflite"
    )  # , quantization_config=config)

    print(model.evaluate_tflite("People_Detection_2.tflite", test_data))

    # Redirects the stdout to a file
    with open("out.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            # This will print the evaluation results to 'out.txt' instead of the standard output
            print(model.evaluate_tflite("People_Detection_2.tflite", test_data))

    model_path = "People_Detection_2.tflite"

    # Load the labels into a list
    classes = ["???"] * model.model_spec.config.num_classes
    label_map = model.model_spec.config.label_map
    for label_id, label_name in label_map.as_dict().items():
        classes[label_id - 1] = label_name

    # Define a list of colors for visualization
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

    local_image_path = "0a80b03afcf13297.jpg"
    detection_threshold = 0.3

    im = Image.open(local_image_path)
    im.thumbnail((512, 512), Image.LANCZOS)
    im.save("saved", "JPEG")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        local_image_path, interpreter, threshold=detection_threshold
    )

    # Show the detection result
    img2 = Image.fromarray(detection_result_image)
    plt.figure(figsize=(10, 10))
    plt.imshow(detection_result_image)
    plt.axis("off")  # Don't show axes for images
    plt.show()
