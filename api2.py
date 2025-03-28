import datetime
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
from deepface import DeepFace
from insightface.app import FaceAnalysis
import mediapipe as mp
import cv2
import random

app = Flask(__name__)
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img1_path = "test.png"

# Connect to MongoDB
URI = "mongodb://localhost:27017/"
client = MongoClient('mongodb://localhost:27017/')
db = client['Liveliness']
User = db['Users']

face = FaceAnalysis(name="buffalo_l")
face.prepare(ctx_id=0, det_size=(640, 640))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# tasks = ["Look Front", "Look Left", "Look Right", "Look Up", "Look Down"]
# selected_task = ""

# Function to decode base64 image from frontend
def decode_image(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error decoding image:", e)
        return None


# this code for postman api testing :-
# def decode_image(image_file):
#     try:
#         image_file.seek(0)  # Ensure file pointer is at the beginning
#         img_array = np.frombuffer(image_file.read(), np.uint8)  # Read bytes as NumPy array
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image
#         if img is None:
#             raise ValueError("cv2.imdecode failed to decode image")  # Explicit error message
#         return img
#     except Exception as e:
#         print("Error decoding image:", str(e))  # Print error for debugging
#         return None


# Function to get stored reference embeddings from MongoDB
def get_reference_embedding(email):
    user = collection.find_one({"email": email})
    if user and "face_embedding" in user:
        return np.array(user["face_embedding"])
    return None

# Function to compute embeddings of captured image
def compute_embedding(img):
    faces = face.get(img)
    if len(faces) > 0:
        return faces[0].normed_embedding
    return None

# # Function to detect head position using MediaPipe
def detect_head_position(image, face_landmarks, img_w, img_h):
    face_3d, face_2d = [], []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
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


# Function to check whether user performed correct task or not
def validate_task(image):
    img_h, img_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x, y, z = detect_head_position(image, face_landmarks, img_w, img_h)
            if y < -10:
                return "Look Left"
            elif y > 10:
                return "Look Right"
            elif x < -10:
                return "Look Down"
            elif x > 10:
                return "Look Up"
            else:
                return "Look Front"
    return "Unknown"

# Function to check for spoofing using DeepFace
def check_liveness(img):
    try:
        result = DeepFace.extract_faces(img_path=img, detector_backend="opencv", enforce_detection=False, align=False, anti_spoofing=True)
        if result and "is_real" in result[0]:
            return "Live" if result[0]["is_real"] else "Spoof"
    except Exception as e:
        print("Liveness detection error:", e)
    return "Unknown"

# registartion route
@app.route("/register", methods=["POST"])
def register():
    image = request.form.get("Image")
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not all([image, name, email, password]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400  # Bad request

    if User.find_one({"email": email}):
        return jsonify({"status": "error", "message": "User already exists"}), 400

    image = image.split(",")[1]
    Timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # save the image from base64 to a file
    os.makedirs("images", exist_ok=True)
    filename = f'images/{Timestamp}.jpeg'
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image))

     # Load the image and extract embeddings
    img = cv2.imread(filename)
    faces = face.get(img)
    
    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    # Extract the first detected face embedding
    face_embedding = faces[0].embedding.tolist()

    User.insert_one({"name": name, "email": email, "password": password, "image": filename,"face_embedding": face_embedding})

    return jsonify({"status": "success"})


''' use random function in frontend to send randomly selected task to backend at /verify route. This one has scalability issues '''

# # route for sending random task to frontend
# @app.route("/task", methods=["GET"])
# def get_random_task():
#     selected_task = random.choice(tasks)
#     return jsonify({"task": selected_task})


# actual verification route
@app.route('/verify', methods=['POST'])
def verify():
    try:  
        '''for postman api testing:-'''
        # selected_task = "Look Front"
        # email = request.form.get("email")
        # img_base64 = request.files.get("image")

        email = request.form.get("email")
        selected_task = request.form.get("task")
        img_base64 = request.form.get("image")
        img_base64 = img_base64.split(",")[1]

        if not email or not img_base64:
            return jsonify({"error": "Missing email or image"}), 400

        # Decode and process the image
        img = decode_image(img_base64)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Get stored reference embedding
        reference_embedding = get_reference_embedding(email)
        if reference_embedding is None:
            return jsonify({"error": "Reference embedding not found for this user"}), 404

        # Compute embeddings for the captured image
        captured_embedding = compute_embedding(img)
        if captured_embedding is None:
            return jsonify({"error": "No face detected in the captured image"}), 400

       # Compute similarity
        def normalize(embedding):
            return embedding / np.linalg.norm(embedding)
        similarity = np.dot(normalize(captured_embedding), normalize(reference_embedding))
        print(similarity)
        threshold = 0.6  # Adjust based on performance

        # Check face match
        face_match = "Matched" if similarity >= threshold else "Not Matched"

        # Check liveness
        liveness_status = check_liveness(img)

        # Task validation
        task_validity = ""
        task_result = validate_task(img)
        if task_result == selected_task:
            task_validity = "Correct"
        else:
            task_validity = "Incorrect"

        # Response
        return jsonify({
            "face_match": face_match,
            "similarity": round(float(similarity), 2),
            "liveness_status": liveness_status,
            "task_validity": task_validity
        })

    except Exception as e:
        # import traceback
        # print(traceback.format_exc())  # Print the full error stack trace
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)
