import datetime
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
from deepface import DeepFace
from insightface.app import FaceAnalysis
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

tasks = ["Look Front", "Look Left", "Look Right", "Look Up", "Look Down"]

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

# Function to get stored reference embeddings from MongoDB
def get_reference_embedding(email):
    user = collection.find_one({"email": email})
    if user and "face_embedding" in user:
        return np.array(user["face_embedding"])
    return None

# Function to compute embeddings of captured image
def compute_embedding(img):
    faces = face_app.get(img)
    if len(faces) > 0:
        return faces[0].normed_embedding
    return None

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


# route for sending random task to frontend
@app.route("/task", methods=["GET"])
def get_random_task():
    selected_task = random.choice(tasks)
    return jsonify({"task": selected_task})


# actual verification route
@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        email = data.get("email")
        img_base64 = data.get("image")

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
        similarity = np.dot(captured_embedding, reference_embedding)
        threshold = 0.6  # Adjust based on performance

        # Check face match
        face_match = "Matched" if similarity > threshold else "Not Matched"

        # Check liveness
        liveness_status = check_liveness(img)

        # Response
        return jsonify({
            "face_match": face_match,
            "similarity": round(float(similarity), 2),
            "liveness_status": liveness_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)
