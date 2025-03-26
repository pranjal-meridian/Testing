import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
import mediapipe as mp
import time
import random
import os

# Path to the face database folder
face_database_folder = r"C:\Users\ISHAN\liveness\face_database2"

# Initialize face recognition (InsightFace) and liveness detection (DeepFace)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize MediaPipe for head pose detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to compute the average face embedding from multiple images
def get_average_embedding(folder_path):
    embeddings = []
    for file in os.listdir(folder_path):
        if file.startswith("Ishan_") and file.endswith(".jpg"):  # Ensuring correct format
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if len(faces) > 0:
                embeddings.append(faces[0].normed_embedding)
            else:
                print(f"Warning: No face detected in {file}")

    if len(embeddings) == 0:
        print("Error: No valid face embeddings found! Check images.")
        return None  # Return None explicitly for error handling

    return np.mean(embeddings, axis=0)

# Load reference embedding
reference_embedding = get_average_embedding(face_database_folder)
if reference_embedding is None:
    print("Error: Could not compute reference embedding. Check images in face_database folder!")
    exit()

# Function to check for spoofing using DeepFace
def check_spoofing(image_path):
    try:
        face_objs = DeepFace.extract_faces(img_path=image_path, detector_backend="opencv", enforce_detection=False,
                                           align=False, anti_spoofing=True)
        if len(face_objs) > 0 and "is_real" in face_objs[0]:
            return "Live" if face_objs[0]["is_real"] else "Spoof"
    except Exception as e:
        print("Error in spoof detection:", e)
    return "Unknown"

# Function to compare faces
def compare_faces(img, reference_embedding, threshold=0.6):
    if reference_embedding is None:
        print("Error: Reference embedding is None. Face database might be empty or invalid.")
        return "Not in Db", None

    faces = app.get(img)
    if len(faces) == 0:
        return "Not in Db", None

    for face in faces:
        similarity = np.dot(face.normed_embedding, reference_embedding)
        if similarity > threshold:
            return "Matched", similarity

    return "Not in Db", None

# Function to detect head position using MediaPipe
def detect_head_position(image, face_landmarks, img_w, img_h):
    """Detects head position based on face landmarks."""
    face_3d, face_2d = [], []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:  # Specific face landmarks
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
    return x, y, z

# Initialize task list - now includes Look Up and Look Down
tasks = ["Look Front", "Look Left", "Look Right", "Look Up", "Look Down"]

# Initialize camera
cap = cv2.VideoCapture(0)

# Main loop for task detection, face matching, and liveness check
for _ in range(3):  # Run for 3 tasks
    task = random.choice(tasks)
    print(f"Task: {task}")
    start_time = time.time()

    last_frame = None  # Variable to store the last frame

    while time.time() - start_time < 5:  # 5-second timer for the task
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = image.shape

        remaining_time = 5 - int(time.time() - start_time)

        # Perform task detection (head pose)
        text = "Forward"  # Default is forward direction
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x, y, z = detect_head_position(image, face_landmarks, img_w, img_h)
                if y < -10:
                    text = "Look Left"
                elif y > 10:
                    text = "Look Right"
                elif x < -10:
                    text = "Look Down"
                elif x > 20:
                    text = "Look Up"
                else:
                    text = "Look Front"

        # Save the last frame for checking after time ends
        last_frame = image

        # Display task, remaining time, liveness, and face match status on the frame
        cv2.putText(image, f"Task: {task}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Time Left: {remaining_time}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Head Position: {text}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Face Recognition & Task Detection', image)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Perform checks after the task time is finished (only on the last frame)
    if last_frame is not None:
        name, similarity = compare_faces(last_frame, reference_embedding)
        liveness_status = check_spoofing(last_frame)

        # Display match and liveness status after task time
        print(f"Face Match: {name} (Similarity: {similarity:.2f})" if similarity else "Face Not Recognized")
        print(f"Liveness Status: {liveness_status}")

        # Check if the task was performed correctly
        if text == task and name == "Matched" and liveness_status == "Live":
            print(f"✅ Correct task performed: {text}")
        else:
            print(f"❌ Incorrect task: {text} or failed authentication!")

cap.release()
cv2.destroyAllWindows()
