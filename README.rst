Custom trained transfer learning model
==================================================================

Steps to create a tflite model thorugh tflite model maker
---------------------------------------------------------

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


Making a tflite model with mediaPipe model
------------------------------------------

This code is used to create a tflite model using the mediapipe model maker. The model is trained on the base model of MobileNetV2. The model is meant to be trained on Google Colab as it is considered optimal to use a GPU for training. 
The dataset needs to be in the pascal VOC format for training and validation. The dataset folder should be in the following format:

.. code-block:: text

    dataset
    ├── train
    │   ├── images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   └── ...
    │   └── annotations
    │       ├── 0001.xml
    │       ├── 0002.xml
    │       └── ...
    └── val
        ├── images
        │   ├── 0001.jpg
        │   ├── 0002.jpg
        │   └── ...
        └── annotations
            ├── 0001.xml
            ├── 0002.xml
            └── ...

Reccomended to use Roboflow to convert the dataset to the pascal VOC format.

Steps to create a tflite model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install the required packages by running the first block.

2. Upload the dataset to Google Colab. The dataset should be uploaded to the 'content' folder in Google Colab.

3. Run the code in the second and third block to create the tflite model.

4. Export and download the tflite model to your local machine.


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

