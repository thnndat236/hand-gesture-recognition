# https://learn.microsoft.com/en-us/windows/wsl/connect-usb
# Powershell:
# usbipd list
# usbipd bind --busid 1-7
# usbipd attach --wsl --busid 1-7
# Bash:
# lsusb
# ls -la /dev/video*

import argparse
import copy
import itertools
import csv
import time
import sys

import numpy as np
import cv2
import mediapipe as mp

# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import ASLClassifier

from utils import draw_landmarks_on_image, draw_bounding_rect, draw_info_text


ASL_CLASSIFIER_LABEL = 'model/asl_classifier/asl_labels.csv'


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width)
        landmark_y = min(int(landmark.y * image_height), image_height)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    
    return [x, y, x + w, y + h]


def calc_landmark_array(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = []
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width)
        landmark_y = min(int(landmark.y * image_height), image_height)
        
        landmark_array.append([landmark_x, landmark_y])

    return landmark_array


def pre_process_landmark(landmark_array):
    temp_landmark_array = copy.deepcopy(landmark_array)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark in enumerate(landmark_array):
        if index == 0:
            base_x, base_y = landmark[0], landmark[1]
        
        temp_landmark_array[index][0] -= base_x
        temp_landmark_array[index][1] -= base_y

    # Convert to a one-dimensional list
    temp_landmark_array = list(itertools.chain.from_iterable(temp_landmark_array))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_array)))

    def normalize_(n):
        return n / max_value
    
    temp_landmark_array = list(map(normalize_, temp_landmark_array))

    return temp_landmark_array


def draw_hand_sign(rgb_image, detection_result, asl_classifier, asl_classifier_labels):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Bounding box calculation
        brect = calc_bounding_rect(annotated_image, hand_landmarks)
        
        # Landmark calculation
        landmark_list = calc_landmark_array(annotated_image, hand_landmarks)

        # Convert landmarks to normalized coordinates 
        pre_processed_landmark_list = pre_process_landmark(landmark_list)

        hand_sign_id = asl_classifier(pre_processed_landmark_list)
        
        annotated_image = draw_bounding_rect(annotated_image, brect)
        annotated_image = draw_info_text(annotated_image,
                                         brect,
                                         handedness,
                                         asl_classifier_labels[hand_sign_id])
    
    return annotated_image


def run(model: str, camera_id: int, width: int, height: int,
        num_hands: int, min_hand_detection_confidence: float, 
        min_hand_presence_confidence: float, min_tracking_confidence: float):
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        num_hands: The maximum number of hands detected by the Hand landmark detector.
        min_hand_detection_confidence: The minimum confidence score for the hand detection to be considered successful in palm detection model.
        min_hand_presence_confidence: The minimum confidence score for the hand presence score in the hand landmark detection model.
        min_tracking_confidence: The minimum confidence score for the hand tracking to be considered successful.
    """
    
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_result_list = []
    
    def visualize_callback(result: vision.HandLandmarkerResult,
                           output_image: mp.Image, timestamp_ms: int):
        result.timestamp_ms = timestamp_ms
        detection_result_list.append(result)

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           num_hands=num_hands,
                                           min_hand_detection_confidence=min_hand_detection_confidence,
                                           min_hand_presence_confidence=min_hand_presence_confidence,
                                           min_tracking_confidence=min_tracking_confidence,
                                           result_callback=visualize_callback)
    detector = vision.HandLandmarker.create_from_options(options)

    # ASL Classifier Model
    asl_classifier = ASLClassifier()

    # Read ASL Labels
    with open(ASL_CLASSIFIER_LABEL, encoding='utf-8-sig') as f:
        asl_classifier_labels = csv.reader(f)
        asl_classifier_labels = [
            row[0] for row in asl_classifier_labels
        ]

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, counter)
        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        if detection_result_list:
            # print(detection_result_list)
            vis_image = draw_landmarks_on_image(current_frame, detection_result_list[0])
            vis_image = draw_hand_sign(vis_image,
                                    detection_result_list[0],
                                    asl_classifier,
                                    asl_classifier_labels)
            cv2.imshow('hand_landmark_detector', vis_image)
            detection_result_list.clear()
        else:
            cv2.imshow('hand_landmark_detector', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='model/hand_landmarker/hand_landmarker.task')
    parser.add_argument(
        "--camera_id", help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
      '--frame_width',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=1280)
    parser.add_argument(
      '--frame_height',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=720)
    parser.add_argument(
      '--num_hands',
      help='The maximum number of hands detected by the Hand landmark detector.',
      required=False,
      type=int,
      default=1)
    parser.add_argument(
      '--min_hand_detection_confidence',
      help='The minimum confidence score for the hand detection to be considered successful in palm detection model.',
      required=False,
      type=float,
      default=0.5)
    parser.add_argument(
      '--min_hand_presence_confidence',
      help='The minimum confidence score for the hand presence score in the hand landmark detection model.',
      required=False,
      type=float,
      default=0.5)
    parser.add_argument(
      '--min_tracking_confidence',
      help='The minimum confidence score for the hand tracking to be considered successful.',
      required=False,
      type=float,
      default=0.5)

    args = parser.parse_args()

    run(args.model, int(args.camera_id), args.frame_width, args.frame_height,
        args.num_hands, args.min_hand_detection_confidence, 
        args.min_hand_presence_confidence, args.min_tracking_confidence)


if __name__ == "__main__":
    main()