from Database import Database
from Detection import Interpreter, Detection
import cv2
from threading import Thread
import queue

frame_queue = queue.Queue(maxsize=10)  # Holds frames for processing
display_queue = queue.Queue(maxsize=10)


def capture_and_enqueue(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        print("Capturing frame")
        cv2.waitKey(1)  # You might need a short delay here


def process_and_enqueue(frame_queue, display_queue, detection_obj):
    while True:
        frame = frame_queue.get()
        if frame is None:  # Use None as a signal to stop the thread
            break
        print("Processing frame")
        processed_frame = detection_obj.analyze(
            frame
        )  # Modify `analyze` to accept and return a frame
        while not display_queue.full():
            try:
                display_queue.get_nowait()  # Remove old frame if exists
            except queue.Empty:
                break
        display_queue.put(processed_frame)
        frame_queue.task_done()


def main():
    cap = cv2.VideoCapture(0)
    # Initialize database
    db_name = "mongodbVSCodePlaygroundDB"
    collection_name = "classes"
    database = Database(db_name, collection_name)
    collection = database.get_collection()

    interpreter = Interpreter(
        "Object_Detection-pi/pre-trained model/tflite-models/ssd_mobilenet.tflite",  # noqa
        "transfer-learning-training/labels-ssd.txt",
    )
    labels = interpreter.load_labels()
    input_details, output_details, height, width = interpreter.get_details()
    interpreter = interpreter.get_interpreter()
    # create a detection object
    detection_obj = Detection(
        labels,
        height,
        width,
        interpreter,
        input_details,
        output_details,
        collection,  # noqa
    )
    # Start threads
    processing_thread = Thread(
        target=process_and_enqueue,
        args=(frame_queue, display_queue, detection_obj),
        daemon=True,
    )
    capture_thread = Thread(
        target=capture_and_enqueue, args=(cap, frame_queue), daemon=True
    )

    processing_thread.start()
    capture_thread.start()
    while True:
        if not display_queue.empty():
            frame_to_display = display_queue.get()
            cv2.imshow("Object Detection", frame_to_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    frame_queue.put(None)  # Signal processing thread to stop
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
