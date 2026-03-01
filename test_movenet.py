import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Load MoveNet Thunder from TF Hub
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256
def resize_to_screen(frame, max_height=800):
    """Resizes frame to a max height while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame
def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    y, x, c = frame.shape
    keep_points = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in keep_points:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (255, 0, 0), -1)
VIDEO_SOURCE = "test\Most_Push_Ups.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Pre-process image
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)

    start = time.perf_counter()
    outputs = module.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()
    end = time.perf_counter()

    fps = 1 / (end - start)

    draw_keypoints(frame, keypoints)
    display_frame = resize_to_screen(frame, max_height=720)
    cv2.putText(display_frame, f"MoveNet FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('MoveNet Pose Estimation', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()