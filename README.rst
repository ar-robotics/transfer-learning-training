Custom trained transfer learning model through tflite model maker
==================================================================

Steps to create a tflite model
------------------------------

1. Install the required packages by running the following command:

.. code-block:: bash

    pip install -r requirements.txt

2. In the images folder, there should be all pictures that are to be used for training. The images should be in 'jpeg' or 'png' format. There should also be a csv file that contains the labels for the images. The csv file should be in the following format:

.. code-block:: csv

    set,image(path),label,xmax,xmin,,,ymax,ymin,
    train,images/0001.jpg,0.32,0.444,,,1,0.123,

3. The csv file can be made by running ``csv_files.py``. This will create a csv file that can be used for training. Make sure you go in the code and change paths to the correct paths for your system.

4. Dynamic running of the code will come later. Run the following command to create a tflite model:

.. code-block:: bash

    python Model_maker_tflite.py

5. Testing with a camera can be done if you change the path to your tflite model and ``labels.txt`` file. Run the following command to test the model:

.. code-block:: bash

    python test_with_camera.py
