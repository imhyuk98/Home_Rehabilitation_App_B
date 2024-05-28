import logging
import time
import os
import cv2
import numpy as np
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

# 비디오 파일 경로 및 설정 값들 직접 지정
video_path = 'videos/test3.mp4'
resolution = '432x368'
model = 'mobilenet_thin'
show_process = False
showBG = True
use_tensorrt = False  # TensorRT 사용 여부

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def initialize_pose_estimator(model, resolution):
    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    model_path = get_graph_path(model)
    if not os.path.isfile(model_path):
        raise Exception(f"Model file does not exist: {model_path}")

    w, h = model_wh(resolution)
    if w == 0 or h == 0:
        raise Exception(f"Invalid resolution: {resolution}")

    e = TfPoseEstimator(model_path, target_size=(w, h), trt_bool=use_tensorrt)
    return e, w, h

if __name__ == '__main__':
    # GPU 설정
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
            logger.debug('GPU is set')
        except RuntimeError as e:
            logger.error(f"Error during GPU setup: {e}")

    try:
        # 포즈 추정기 초기화
        pose_estimator, w, h = initialize_pose_estimator(model, resolution)

        # 비디오 캡처 초기화
        if not os.path.isfile(video_path):
            raise Exception(f"Video file does not exist: {video_path}")
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise Exception(f"Error opening video stream or file: {video_path}")

        # 웹캠 캡처 초기화
        webcam_cap = cv2.VideoCapture(0)
        if not webcam_cap.isOpened():
            raise Exception(f"Error opening webcam stream")

        while video_cap.isOpened() and webcam_cap.isOpened():
            ret_val_video, video_frame = video_cap.read()
            ret_val_webcam, webcam_frame = webcam_cap.read()

            if not ret_val_video and not ret_val_webcam:
                logger.debug('No frame retrieved from either video or webcam')
                break

            if video_frame is not None and video_frame.size != 0:
                video_frame = cv2.resize(video_frame, (w, h))
                try:
                    humans_video = pose_estimator.inference(video_frame, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
                    if not showBG:
                        video_frame = np.zeros(video_frame.shape)
                    video_frame = TfPoseEstimator.draw_humans(video_frame, humans_video, imgcopy=False)

                    cv2.putText(video_frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('Video Pose Estimation', video_frame)
                except Exception as e:
                    logger.error(f"Error during video pose estimation: {e}")
                    break

            if webcam_frame is not None and webcam_frame.size != 0:
                webcam_frame = cv2.resize(webcam_frame, (w, h))
                try:
                    humans_webcam = pose_estimator.inference(webcam_frame, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
                    if not showBG:
                        webcam_frame = np.zeros(webcam_frame.shape)
                    webcam_frame = TfPoseEstimator.draw_humans(webcam_frame, humans_webcam, imgcopy=False)

                    cv2.putText(webcam_frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('WebCam Pose Estimation', webcam_frame)
                except Exception as e:
                    logger.error(f"Error during webcam pose estimation: {e}")
                    break

            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

        video_cap.release()
        webcam_cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
logger.debug('finished+')
