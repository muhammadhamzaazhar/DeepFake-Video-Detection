# import cv2
# import os
# from mtcnn import MTCNN

# def detect_faces_in_video(video_path, output_dir, padding_percentage=0.3):
#     """
#     Detect faces in a video using MTCNN and save cropped face regions.

#     Args:
#         video_path (str): Path to the input video.
#         output_dir (str): Directory to save the cropped face frames.
#         padding_percentage (float): Percentage of padding to add around the detected face.

#     Returns:
#         list: Paths to the saved cropped face frames.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     detector = MTCNN()
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"Error: Unable to open video file {video_path}")
#         return []

#     frame_count = 0
#     cropped_faces = []
#     valid_video = True  # Assume video is valid unless proven otherwise

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame is None:
#             print(f"Warning: Empty frame at {frame_count}")
#             continue

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = detector.detect_faces(rgb_frame)

#         # Skip video if multiple faces are detected
#         '''
#         if len(faces) > 1:
#             print(f"Skipping video {video_path} due to multiple faces detected.")
#             valid_video = False
#             break
#         '''

#         for i, face in enumerate(faces):
#             keypoints = face['keypoints']
#             if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
#                 print(f"Skipping video {video_path} due to missing eye detection.")
#                 valid_video = False
#                 break

#             confidence = face['confidence']
#             if confidence < 0.85:
#                 continue

#             x, y, w, h = face['box']
#             if w < 50 or h < 50:
#                 continue

#             padding = max(1, int(min(w, h) * padding_percentage))
#             x1 = max(0, x - padding)
#             y1 = max(0, y - padding)
#             x2 = min(rgb_frame.shape[1], x + w + padding)
#             y2 = min(rgb_frame.shape[0], y + h + padding)

#             cropped_face = frame[y1:y2, x1:x2]
#             resized_cropped_face = cv2.resize(cropped_face, (224, 224))

#             face_filename = f"frame_{frame_count:05d}_face_{i}.png"
#             face_path = os.path.join(output_dir, face_filename)
#             cv2.imwrite(face_path, resized_cropped_face)

#             cropped_faces.append(face_path)

#         frame_count += 1

#     cap.release()

#     # If video was invalid, remove all saved frames and return empty list
#     if not valid_video:
#         for face_path in cropped_faces:
#             if os.path.exists(face_path):
#                 os.remove(face_path)
#         return []

#     return cropped_faces

import cv2
import os
from mtcnn import MTCNN

def detect_faces_in_video(video_path, output_dir, padding_percentage=0.3, full_detection_interval=10):
    os.makedirs(output_dir, exist_ok=True)

    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    frame_count = 0
    cropped_faces = []
    valid_video = True
    multi_face_counter = 0
    max_multi_face_frames = 3
    trackers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            print(f"Warning: Empty frame at {frame_count}")
            continue

        if frame_count % full_detection_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)
            trackers = []  

            # if len(faces) > 1:
            #     print(f"Skipping video {video_path} due to multiple faces detected.")
            #     valid_video = False
            #     break

            if len(faces) != 1:
                multi_face_counter += 1
                if multi_face_counter > max_multi_face_frames:
                    print(f"Skipping video {video_path} due to inconsistent face count.")
                    valid_video = False
                    break
                continue

            for i, face in enumerate(faces):
                keypoints = face['keypoints']
                if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
                    print(f"Skipping video {video_path} due to missing eye detection.")
                    valid_video = False
                    break

                confidence = face['confidence']
                if confidence < 0.85:
                    continue

                x, y, w, h = face['box']
                if w < 50 or h < 50:
                    continue

                padding = max(1, int(min(w, h) * padding_percentage))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)

                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue

                resized_cropped_face = cv2.resize(cropped_face, (224, 224))
                face_filename = f"frame_{frame_count:05d}_face_{i}.png"
                face_path = os.path.join(output_dir, face_filename)
                cv2.imwrite(face_path, resized_cropped_face)
                cropped_faces.append(face_path)

                tracker = cv2.TrackerCSRT_create()  
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
        else:
            for i, tracker in enumerate(trackers):
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    padding = max(1, int(min(w, h) * padding_percentage))
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    cropped_face = frame[y1:y2, x1:x2]
                    if cropped_face.size == 0:
                        continue

                    resized_cropped_face = cv2.resize(cropped_face, (224, 224))
                    face_filename = f"frame_{frame_count:05d}_face_{i}.png"
                    face_path = os.path.join(output_dir, face_filename)
                    cv2.imwrite(face_path, resized_cropped_face)
                    cropped_faces.append(face_path)

        frame_count += 1

    cap.release()

    if not valid_video:
        for face_path in cropped_faces:
            if os.path.exists(face_path):
                os.remove(face_path)
        return []

    return cropped_faces