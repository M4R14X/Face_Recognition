from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import os
import time
import threading
import numpy as np
import pickle

app = Flask(__name__)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
cap = cv2.VideoCapture(0)
capturing = False
person_name = ""
image_limit = 50
dataset_path = "dataset"
model_path = "face_model.yml"
labels_path = "labels.pickle"

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/choose')
def choose():
    return render_template('choose.html')

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == 'POST':
        username = request.form['username']
        # Redirect to face capturing with the provided name
        return redirect(url_for('capture_faces', username=username))
    return render_template('name.html')

# Route: Capture and Train Page
@app.route('/capture', methods=['POST'])
def capture():
    global person_name, image_limit
    person_name = request.form['name']
    image_limit = int(request.form.get('limit', 50))
    return render_template('capture.html', name=person_name)
# Route: Start Webcam Feed
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: Capture and Train Images
@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing, person_name, image_limit
    person_path = os.path.join(dataset_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Start capturing in a new thread
    capturing = True
    threading.Thread(target=save_images, args=(person_path,)).start()
    return jsonify({"status": "Capturing started!"})

def save_images(person_path):
    global capturing
    count = 0

    while capturing and count < image_limit:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            count += 1
            file_name = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_name, face)

        if count >= image_limit:
            capturing = False
            
# Route: Train Model
@app.route('/train', methods=['POST'])
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_dict = {}
    label_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[label_id] = person_name
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(image)
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    with open(labels_path, 'wb') as f:
        pickle.dump(label_dict, f)

    return jsonify({"status": "Training completed!"})

# Route: Recognize Page
@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, 'rb') as f:
        label_dict = pickle.load(f)

    def gen():
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                id_, confidence = recognizer.predict(face)
                if confidence < 50:  # Confidence threshold
                    name = label_dict[id_]
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
