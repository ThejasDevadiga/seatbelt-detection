import sys
import os
import yaml
import logging
import cv2  # type: ignore
import uuid
import time
import tritonclient.grpc as grpcclient  # type: ignore
import tritonclient.http as httpclient  # type: ignore
from utils.processing import preprocess,postprocess
from utils.labels import ObjectDetectionLabels
from utils.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
import numpy as np

logger = logging.getLogger(__name__)


GRPC_URL = 'localhost:8001'
HTTP_URL = 'localhost:8000'

PERSON_CLASSES = {
    ObjectDetectionLabels.PERSON.value,
}

CATEGORY_COLORS = {
    "person": (255, 0, 0),    # Blue
    "other":(0,255,0)
}

WIDTH = 640
HEIGHT = 640



def create_triton_http_client( http_url, model_name):
    try:
        logger.info(f"Connecting to Triton server ")
        http_client = httpclient.InferenceServerClient(url=http_url)

        if not http_client.is_server_live():
            logger.error("Triton HTTP server is not live")
            raise RuntimeError("Triton HTTP server is not live")
        if not http_client.is_server_ready():
            logger.error("Triton HTTP server is not ready")
            raise RuntimeError("Triton HTTP server is not ready")
        logger.info("Triton HTTP server is ready")

        if not http_client.is_model_ready(model_name):
            logger.error(f"Model {model_name} is not ready")
            raise RuntimeError(f"Model {model_name} is not ready")

        return  http_client

    except Exception as e:
        logger.exception("Triton client setup failed")
        sys.exit(1)


def create_triton_grpc_client(grpc_url, model_name):
    try:
        logger.info(f"Connecting to Triton GRPC server ")
        grpc_client = grpcclient.InferenceServerClient(url=grpc_url)

        if not grpc_client.is_server_live():
            logger.error("Triton GRPC server is not live")
            raise RuntimeError("Triton GRPC server is not live")
        if not  grpc_client.is_server_ready():
            logger.error("Triton GRPC server is not ready")
            raise RuntimeError("Triton GRPC server is not ready")
        logger.info("Triton GRPC server is ready")

        if not grpc_client.is_model_ready(model_name):
            logger.error(f"Model {model_name} is not ready")
            raise RuntimeError(f"Model {model_name} is not ready")

        return grpc_client

    except Exception as e:
        logger.exception("Triton client setup failed")
        sys.exit(1)


def process_frame(frame, client, model_name, inputs, outputs, width, height):
    try:
        """Process a single frame and return the result"""
        input_image_buffer = preprocess(frame, [width, height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)

        start_time = time.time()
        results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        end_time = time.time()

        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")

        detected_objects = postprocess(
            num_dets, det_boxes, det_scores, det_classes,
            frame.shape[1], frame.shape[0], [width, height]
        )


        for box in detected_objects:
            label = ObjectDetectionLabels(box.classID).name
            if box.classID in PERSON_CLASSES:
                category = "person"
            else:
                category = "Other"

            logger.debug(f"{category} - {label}: {box.confidence:.2f}")

            color = CATEGORY_COLORS[category]
            frame = render_box(frame, box.box(), color=color)
            size = get_text_size(frame, f"{category}", normalised_scaling=0.4)
            frame = render_filled_box(frame, (box.x1, box.y1, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            frame = render_text(frame, f"{category}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.4)

        display_width, display_height = 1280, 400
        h, w = frame.shape[:2]
        if w > h:
            new_w, new_h = display_width, int(h * (display_width / w))
        else:
            new_h, new_w = display_height, int(w * (display_height / h))

        resized_frame = cv2.resize(frame, (new_w, new_h))
        time.sleep(0.001)
        return resized_frame
    except Exception as e:
        logger.exception(f"Error processing frame for {model_name}: {str(e)}")
        time.sleep(0.1)
        return frame


class TritonClient:
    def __init__(self, client_type):
        self.model_key = client_type
        self.client_id = str(uuid.uuid4()) 
        print(f"model_key: {self.model_key}, client_id: {self.client_id}")
        self.model_name = ""
        self.models = {
            "object_detection":{
                "model_name": "object_detection",
                "client_type":"grpc",
                "frame_rate": 30,
            },
           
            "seatbelt_detection":{
                "model_name": "seatbelt",
                "frame_rate": 30,
                "client_type":"grpc",
            },
        }
        self.triton_client = None
        # self.model_name = self.models[self.model_key]["model_name"] 
        # print(f"model_name: {self.model_name}")
        # client_type = self.models[self.model_key]["client_type"] 
        # if client_type=="http":
        #     print("selected: http")
        #     self.triton_client = create_triton_http_client(HTTP_URL, self.model_name)
        # else:
        #     print("selected: grpc")
        #     self.triton_client = create_triton_grpc_client(GRPC_URL, self.model_name)

    def setup_client(self):
        try:
            if self.triton_client is None:
                self.model_name = self.models[self.model_key]["model_name"] 
                print(f"model_name: {self.model_name}")
                client_type  = self.models[self.model_key]["client_type"] 
                if client_type=="http":
                    print("selected: http")
                    self.triton_client = create_triton_http_client(HTTP_URL,self.model_name)
                else:
                    print("selected: grpc")
                    self.triton_client = create_triton_grpc_client(GRPC_URL,self.model_name)
        except Exception as e:
            logger.exception(f"Error setting up Triton client for {self.model_key}: {e}")
            sys.exit(1)
    def start_inference(self):
        if  self.triton_client:
            try:
                width=640
                height=640
                INPUT_NAMES = ["images"]
                OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
                inputs = [grpcclient.InferInput(INPUT_NAMES[0], [1, 3, width, height], "FP32")]
                outputs = [grpcclient.InferRequestedOutput(name) for name in OUTPUT_NAMES]
    
                camera_index = 0
                cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    logger.warning(f"Camera index {camera_index} might be in use or unavailable.")
                    return None
                
                if not cap.isOpened():
                    logger.error(f"FAILED: cannot open video {camera_index}")
                    return

                logger.info(f"Invoking inference using camera: {camera_index}")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to fetch next frame")
                        break
                        
                    frame =  process_frame(frame, self.triton_client, self.model_name, inputs, outputs, width, height)
                    cv2.imshow("GRPC OUTPUT",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                logger.exception(f"Error processing frame for {self.model_key}: {e}")
                time.sleep(0.1)

if __name__ == '__main__':

    object_detection_client = TritonClient("object_detection")
    object_detection_client.setup_client()
    object_detection_client.start_inference()

    seatbelt_detection_client = TritonClient("seatbelt_detection")
    seatbelt_detection_client.setup_client()
    seatbelt_detection_client.start_inference()