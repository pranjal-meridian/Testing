import cv2
import numpy as np
from insightface.app import FaceAnalysis
import mediapipe as mp
import time

# Initialize Face Recognition
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize MediaPipe for Head Pose Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to detect head pose
def detect_head_position(image, face_landmarks, img_w, img_h):
    face_3d, face_2d = [], []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

    return x, y, z

# Capture images for Left, Right, Front, Up, and Down poses
cap = cv2.VideoCapture(0)
poses = {"Look Left": None, "Look Right": None, "Look Front": None, "Look Up": None, "Look Down": None}
embeddings = []

for pose in poses.keys():
    print(f"Task: {pose} - Hold still for 5 seconds!")
    start_time = time.time()
    last_frame = None

    while time.time() - start_time < 5:
        success, image = cap.read()
        if not success:
            continue

        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        head_position = "Unknown"
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x, y, z = detect_head_position(image, face_landmarks, img_w, img_h)

                if y < -10:
                    head_position = "Look Left"
                elif y > 10:
                    head_position = "Look Right"
                elif x < -10:
                    head_position = "Look Down"
                elif x > 20:
                    head_position = "Look Up"
                else:
                    head_position = "Look Front"

        remaining_time = 5 - int(time.time() - start_time)
        cv2.putText(image, f"Task: {pose}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(image, f"Time Left: {remaining_time}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Detected: {head_position}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Head Pose Capture', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        last_frame = image if head_position == pose else last_frame

    # Capture the final frame
    if last_frame is not None:
        poses[pose] = last_frame
        faces = app.get(last_frame)
        if len(faces) > 0:
            embeddings.append(faces[0].normed_embedding)

cap.release()
cv2.destroyAllWindows()

# Compute and Print Average Embedding
if len(embeddings) > 0:
    avg_embedding = np.mean(embeddings, axis=0)
    print("Average Embedding:", avg_embedding)
else:
    print("Error: No valid face embeddings found! Check if face was detected correctly.")
