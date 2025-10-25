import os
import sys
import time
import cv2 # type: ignore
import csv
import signal
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from threading import Thread
from Yolov7_inference import InferenceResult
from queue import Queue
from threading import Thread

csv_queue = Queue()

#  --------------------------------------------------------------------
#  --------------------------------------------------------------------
MODEL_PATH = "model/"
MODEL_NAME = "default_model"
MODEL_EXT = ".pt"
CONF_THRESHOLD = 0.7
CAMERA_INDEX = 0
TARGET_FPS = 30
INPUT_SIZE = 416 
LOG_FILE = "logs/predictions.log"

#  --------------------------------------------------------------------
#  --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WebcamPredictor")

#  --------------------------------------------------------------------
#  --------------------------------------------------------------------
try:
    from Yolov7_inference import Predictor
except ImportError as e:
    logger.error("Failed to import Predictor class: %s", e)
    sys.exit(1)


class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.stream.read()
        if not self.ret:
            raise RuntimeError("Failed to read from webcam")
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.stream.release()

# ---------------------------------------------------
# ---------------------------------------------------
model_file = Path(MODEL_PATH) / f"{MODEL_NAME}{MODEL_EXT}"
if not model_file.exists():
    logger.error(" file not found: %s", model_file)
    sys.exit(1)


try:
    prediction_handler = Predictor(str(model_file), "cpu")
    if prediction_handler.model_loaded is False:
        logger.error("Model loading failed for: %s", model_file)
        sys.exit(1)
    prediction_handler.image_size = INPUT_SIZE
    logger.info("Model loaded   %s", model_file)
except Exception as e:
    logger.exception("Failed to initialize : %s", e)
    sys.exit(1)

# Cleanup
def cleanup_and_exit(signum=None, frame=None):
    global csv_thread
    logger.info("Shutting down...")
    global cap
    if 'cap' in globals() and isinstance(cap, WebcamStream):
        cap.stop()
    csv_queue.put(None)
    csv_thread.join()

    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


def csv_writer_thread(csv_path):
    with csv_path.open("a", newline='') as file:
        writer = csv.writer(file)
        while True:
            item = csv_queue.get()
            if item is None:
                break
            timestamp, detections = item
            for det in detections:
                writer.writerow([
                    timestamp,
                    det['label'],
                    det['confidence'],
                    det['min_x'],
                    det['min_y'],
                    det['max_x'],
                    det['max_y'],
                    det['seatbelt_label'],
                    det['seatbelt_confidence']  
                ])
            file.flush()


def main():
    global cap
    INPUT_VIDEO = "/home/thejas-devadiga/Videos/test_2.mp4"
    # cap = WebcamStream(CAMERA_INDEX)
    cap = cv2.VideoCapture(INPUT_VIDEO)

    logger.info("Camera stream started.")
    frame_delay = 1.0 / TARGET_FPS
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "detections_log.csv"

    csv_thread = Thread(target=csv_writer_thread, args=(csv_path,), daemon=True)
    csv_thread.start()

    frame_count = 0
    last_log_time = time.time()

    while True:
            ret, frame = cap.read()
            # frame = cv2.imread("src/without-seatbelt.png")
            # if not ret:
            #     logger.warning("Failed to grab frame.")
            #     continue
            start_time = time.time()
            try:
                result = prediction_handler.detect(frame, CONF_THRESHOLD)

                for det in result['detections']:
                    cv2.rectangle(frame, (det['min_x'], det['min_y']),(det['max_x'], det['max_y']), det["color"], 2)

                    cv2.putText(frame, f"{det['label']}:{det['confidence']:.2f}",
                                (det['min_x'], det['min_y'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, det["color"], 2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_queue.put((timestamp, result['detections']))

                cv2.imshow("Feed", frame)
                cv2.waitKey(1)
                frame_count += 1
                if frame_count % 10 == 0:
                    logger.info(f"Frame {frame_count} | Processing time: {result['total_time']}")
                
            except Exception as e:
                logger.exception("Error during prediction: %s", e)
                continue

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.2f}")

            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     logger.info("Exit requested by user.")
            #     break

    cleanup_and_exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unexpected error in main loop: %s", e)
        cleanup_and_exit()
