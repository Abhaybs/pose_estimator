from ultralytics import YOLO
import cv2
import time

# Use the 'n' (nano) version for best speed on local systems
def resize_to_screen(frame, max_height=800):
    """Resizes frame to a max height while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame
model = YOLO('yolov8n-pose.pt') 
VIDEO_SOURCE = "test\Most_Push_Ups.mp4" # Use 0 for webcam, or provide a video file path
cap = cv2.VideoCapture(VIDEO_SOURCE)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    start = time.perf_counter()
    # stream=True is more memory efficient for video
    results = model(frame, stream=False, verbose=False) 
    end = time.perf_counter()

    fps = 1 / (end - start)

    # Plot results on the frame
    annotated_frame = results[0].plot()
    display_frame = resize_to_screen(annotated_frame, max_height=720)
    cv2.putText(display_frame, f"YOLOv8 FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('YOLOv8 Pose Estimation', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()