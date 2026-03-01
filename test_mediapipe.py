import cv2
import mediapipe as mp
import time
from pathlib import Path
from urllib.request import urlretrieve


def draw_pose_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def run_legacy_solutions(cap):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        end = time.perf_counter()

        fps = 1 / (end - start)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"MediaPipe FPS: {fps:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("MediaPipe Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_tasks_api(cap):
    model_path = Path("pose_landmarker_lite.task")
    if not model_path.exists():
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        print("Downloading pose model...")
        urlretrieve(model_url, model_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)
            end = time.perf_counter()

            fps = 1 / (end - start)

            if result.pose_landmarks:
                draw_pose_landmarks(frame, result.pose_landmarks[0])

            cv2.putText(
                frame,
                f"MediaPipe FPS: {fps:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("MediaPipe Pose Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
VIDEO_SOURCE = "test\Most_Push_Ups_in_1_MINUTE_WORLD_RECORD_720P.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

try:
    if hasattr(mp, "solutions"):
        run_legacy_solutions(cap)
    else:
        run_tasks_api(cap)
except FileNotFoundError as error:
    print(error)

cap.release()
cv2.destroyAllWindows()