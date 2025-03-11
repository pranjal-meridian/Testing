import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
from insightface.app import FaceAnalysis
import cv2

app = Flask(__name__)
img1_path = "img.png"

face = FaceAnalysis(name="buffalo_l")
face.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    faces = face.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding


existing_face_embedding = get_face_embedding("img.png")


def compare_faces(img, reference_embedding, threshold=0.6):
    faces = face.get(img)
    if len(faces) == 0:
        return None, None

    for facee in faces:
        similarity = np.dot(facee.normed_embedding, reference_embedding)
        print("Similarity:", similarity)
        if similarity > threshold:
            return "name from Db", similarity

    return "Not in Db", None


def check_spoofing(frame):
    try:
        face_objs = DeepFace.extract_faces(img_path=frame, detector_backend="opencv", enforce_detection=False,
                                           align=False, anti_spoofing=True)
        if len(face_objs) > 0 and "is_real" in face_objs[0]:
            return "Live" if face_objs[0]["is_real"] else "Spoof"
    except Exception as e:
        print("Error in spoof detection:", e)
    return "Unknown"


@app.route("/verify", methods=["POST"])
def verify():
    print(request.files)
    # use insideface to detect the face
    image = request.files["image"]
    image.save("test.png")
    frame = cv2.imread("test.png")

    name, similarity = compare_faces(frame, existing_face_embedding)
    # use deepface to verify spoofing
    status = check_spoofing(frame)
    return jsonify({"name": name, "similarity": str(similarity), "status": status})


if __name__ == "__main__":
    app.run(debug=True)
