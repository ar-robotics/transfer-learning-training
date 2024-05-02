import os
import cv2
import numpy as np
import tensorflow as tf

# import argparse

# # import contextlib
# from PIL import Image
# from matplotlib import pyplot as plt
from absl import logging

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

# shows only errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""check tensorflow version"""
tf.get_logger().setLevel("ERROR")

logging.set_verbosity(logging.ERROR)


class DataLoader:
    def __init__(self, csv_file_path, images_dir):
        self.csv_file_path = csv_file_path
        self.images_dir = images_dir

    def load_data(self):
        """Loads data from the csv file.
        Args:
         csv_file_path: The file path to the csv file.
         images_dir: The directory that contains images.
        Returns:
        A Dataloader containing the data of training, validation and test set
        """
        train_data, test_data, validation_data = (
            object_detector.DataLoader.from_csv(  # noqa
                self.csv_file_path, images_dir=self.images_dir
            )
        )
        return train_data, test_data, validation_data


class ModelTrainer:
    """A class to train the model on the given dataset and evaluate it.

    Attributes:
        train_data: Training dataset, in the form of Dataloader.
        validation_data: Validation dataset, in the form of Dataloader.
        eval: The evaluation result.
        model: The trained model.
    """

    def __init__(self, train_data, validation_data):
        """Initializes the ModelTrainer with the given dataset.

        Args:
            train_data: Training dataset, in the form of Dataloader.
            validation_data: Validation dataset, in the form of Dataloader.
        """
        self.train_data = train_data
        self.validation_data = validation_data
        self.eval = None
        self.model = None

    def train_model(self):
        """Fine tunes the EfficientDet-Lite0 model on the given dataset.

        Args:
            train_data: Training dataset, in the form of Dataloader.
            validation_data: Validation dataset, in the form of Dataloader.

        Returns:
            A trained model.
        """
        spec = model_spec.get("efficientdet_lite0")
        self.model = object_detector.create(
            self.train_data,
            model_spec=spec,
            batch_size=4,
            train_whole_model=True,
            epochs=25,
            validation_data=self.validation_data,
        )
        return self.model

    def evaluate_and_export(
        self,
        test_data,
    ):
        """Evaluates the model and exports it to the export directory.

        Args:
            test_data: Test dataset, in the form of Dataloader.
        """
        self.eval = print(self.model.evaluate(test_data))
        self.model.export(export_dir=".")
        self.model.export(
            export_dir=".",
            export_format=[ExportFormat.TFLITE, ExportFormat.LABEL],  # noqa
        )
        # self.model.export(export_dir=".", tflite_filename="People-det.tflite")  # noqa

        # self.model.export(
        #     export_dir="tflite_models",
        #     tflite_filename="f16-People-det.tflite",
        #     quantization_config=QuantizationConfig.for_float16(),
        # )
        self.model.export(
            export_dir="tflite_models",
            tflite_filename="i8-People-det.tflite",
            quantization_config=QuantizationConfig.for_int8((self.train_data)),
        )


class ObjectDetector:
    """A class to perform object detection with a given model.

    Attributes:
        model: The trained model.
        model_path: The file path to the trained model.
        classes: A list of class labels.
        colors: A list of colors for visualization.
        interpreter: The TensorFlow Lite interpreter.
    """

    def __init__(self, model, model_path, classes):
        # Define a list of colors for visualization
        self.colors = np.random.randint(
            0, 255, size=(len(classes), 3), dtype=np.uint8
        )  # noqa
        self.model_path = model_path
        self.model = model
        self.classes = classes
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def preprocess_image(self, image_path, input_size):
        return
        # Implementation remains the same as the function above

    def detect_objects(self, image, threshold=0.5):
        return
        # Implementation remains the same as the function above

    def run_odt_and_draw_results(self, image_path, interpreter, threshold=0.5):
        """Run object detection on the input image and draw the results.

        Args:
            image_path: The file path to the input image.
            interpreter: The TensorFlow Lite interpreter.
            threshold: The minimum confidence score for detected objects.

        Returns:
            A NumPy array of the input image with the detection results.
        """
        # Load the input shape required by the model
        _, input_h, input_w, _ = interpreter.get_input_details()[0]["shape"]

        # Load the input image and preprocess it
        preprocessed_image, original_image = self.preprocess_image(
            image_path, (input_h, input_w)
        )

        # Run object detection on the input image
        results = self.detect_objects(
            interpreter, preprocessed_image, threshold=threshold
        )

        # Plot the detection results on the input image
        original_image_np = original_image.numpy().astype(np.uint8)
        for obj in results:
            # Convert the object bounding box from relative coordinates to
            # absolute coordinates based on the original image resolution
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(xmin * original_image_np.shape[1])
            xmax = int(xmax * original_image_np.shape[1])
            ymin = int(ymin * original_image_np.shape[0])
            ymax = int(ymax * original_image_np.shape[0])

            # Find the class index of the current object
            class_id = int(obj["class_id"])

            # Draw the bounding box and label on the image
            color = [int(c) for c in self.colors[class_id]]
            cv2.rectangle(
                original_image_np, (xmin, ymin), (xmax, ymax), color, 2
            )  # noqa
            # Make adjustments to make the label visible for all objects
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.0f}%".format(
                self.classes[class_id], obj["score"] * 100
            )  # noqa

            cv2.putText(
                original_image_np,
                label,
                (xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Return the final image
        original_uint8 = original_image_np.astype(np.uint8)
        return original_uint8


if __name__ == "__main__":
    csv_file_path = "transfer-learning-training/csv_files/_annotations.csv"
    images_dir = "valid"  # noqa
    data_loader = DataLoader(csv_file_path, images_dir)
    train_data, test_data, validation_data = data_loader.load_data()
    model_trainer = ModelTrainer(train_data, validation_data)
    model = model_trainer.train_model()
    model_trainer.evaluate_and_export(test_data)

    # model_path = "tflite_models/model.tflite"

    # local_image_path = "transfer-learning-training/iii.jpg"
    # detection_threshold = 0.5

    # im = Image.open(local_image_path)
    # im.thumbnail((512, 512), Image.LANCZOS)
    # im.save("saved", "JPEG")

    # # Load the TFLite model
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    # # Load the labels into a list
    # classes = ["???"] * model.model_spec.config.num_classes
    # label_map = model.model_spec.config.label_map
    # for label_id, label_name in label_map.as_dict().items():
    #     classes[label_id - 1] = label_name

    # object_d = ObjectDetector(model, model_path, classes)

    # # Run inference and draw det result on the local copy of the og image
    # detection_result_image = object_d.run_odt_and_draw_results(
    #     local_image_path, interpreter, threshold=detection_threshold
    # )

    # # Show the detection result
    # img2 = Image.fromarray(detection_result_image)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(detection_result_image)
    # plt.axis("off")  # Don't show axes for images
    # plt.show()
