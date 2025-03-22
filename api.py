import datetime
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
from deepface import DeepFace
from insightface.app import FaceAnalysis
import cv2

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

    User.insert_one({"name": name, "email": email, "password": password, "image": filename})

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)
