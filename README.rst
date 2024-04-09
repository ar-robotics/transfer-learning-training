Custom trained transfer learning model through tflite model maker
==================================================================

Steps to create a tflite model
------------------------------

1. Install the required packages by running the following command:

.. code-block:: bash

    pip install -r requirements.txt

2. In the images folder, there should be all pictures that are to be used for training. The images should be in 'jpeg' or 'png' format. There should also be a csv file that contains the labels for the images. The csv file should be in the following format:

.. code-block:: text

    set,image(path),label,xmax,xmin,,,ymax,ymin,
    train,images/0001.jpg,0.32,0.444,,,1,0.123,

3. The csv file can be made by running ``csv_files.py``. This will create a csv file that can be used for training. Make sure you go in the code and change paths to the correct paths for your system.

4. Dynamic running of the code will come later. Run the following command to create a tflite model:

.. code-block:: bash

    python Model_maker_tflite.py

5. Testing with a camera can be done if you change the path to your tflite model and ``labels.txt`` file. Run the following command to test the model:

.. code-block:: bash

    python test_with_camera.py

=======================================
 Webcam Object Detection Using TensorFlow
=======================================

Description
===========

This program utilizes a TensorFlow Lite model to perform object detection on a live webcam feed. It draws boxes and scores around the objects of interest in each frame from the webcam. To improve frames per second (FPS), the webcam object runs in a separate thread from the main detection algorithm.
Inspiration for the Videostream is from 
Installation
============

To run this project, ensure you have Python version 3.8 or above installed on your system. Clone the project repository to your local machine and navigate to the project directory.

Requirements
============

Before running the program, you need to install the required Python libraries. You can do this by running:

.. code-block:: bash

    pip install -r requirements.txt


Usage
=====

To run the object detection program, execute the following command in the terminal:

.. code-block:: bash

    python main.py --modeldir YOUR_MODEL_DIRECTORY

Where `YOUR_MODEL_DIRECTORY` is the path to the directory containing your TensorFlow Lite model file and label map.

Command-Line Arguments
----------------------

- `--modeldir`: Folder where the `.tflite` file is located (required).
- `--graph`: Name of the `.tflite` file, default is `detect.tflite`.
- `--labels`: Name of the label map file, default is `labelmap.txt`.
- `--threshold`: Minimum confidence threshold for displaying detected objects, default is 0.5.
- `--resolution`: Desired webcam resolution in WxH. Ensure the webcam supports the resolution, default is `1280x720`.

Files and Modules
=================

- `main.py`: The main script that initiates the webcam feed and object detection.
- `Videostream.py`: Handles video streaming from the webcam in a separate thread to improve performance.
- `Detection.py`: Contains the `ExtractModel` and `Detection` classes for loading the TensorFlow Lite model and performing object detection.
- `Database.py`: Manages database operations, including connecting to MongoDB, inserting data, and querying data.







Documentation
-------------

HTML
^^^^

.. code-block:: bash

    sudo apt install make python3-sphinx python3-pip
    pip3 install furo sphinxcontrib-jquery
    cd docs/
    make clean 
    make html

PDF
^^^

.. code-block:: bash

    sudo apt install latexmk texlive-latex-extra
    cd docs/
    make latexpdf
