import logging
import time
import os
import cv2
import numpy as np
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
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
        if not os.path.isfile(video_path):
            raise Exception(f"Video file does not exist: {video_path}")

        logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        model_path = get_graph_path(model)
        if not os.path.isfile(model_path):
            raise Exception(f"Model file does not exist: {model_path}")

        w, h = model_wh(resolution)
        if w == 0 or h == 0:
            raise Exception(f"Invalid resolution: {resolution}")

        e = TfPoseEstimator(model_path, target_size=(w, h), trt_bool=use_tensorrt)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Error opening video stream or file: {video_path}")

        frame_count = 0  # 프레임 카운터 추가
        display_interval = 10  # 좌표값을 출력할 프레임 간격 설정
        
        while cap.isOpened():
            ret_val, image = cap.read()
            frame_count += 1
            logger.debug(f"Processing frame {frame_count}")

            if not ret_val:
                logger.debug('No frame retrieved from video stream')
                break

            # 이미지 크기가 0인 경우 예외 처리 추가
            if image is None or image.size == 0:
                logger.error('Empty frame encountered')
                continue

            try:
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
                if not showBG:
                    image = np.zeros(image.shape)
                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                # 프레임 간격에 따라 좌표값 출력
                if frame_count % display_interval == 0:
                    for human in humans:
                        for i in range(len(human.body_parts)):
                            body_part = human.body_parts[i]
                            logger.info(f'Body part {i}: ({body_part.x}, {body_part.y})')

                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('tf-pose-estimation result', image)
                fps_time = time.time()
                if cv2.waitKey(1) == 27:
                    break
            except Exception as e:
                logger.error(f"Error during pose estimation: {e}")
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
logger.debug('finished+')
