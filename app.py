from flask import Flask, redirect, render_template, request, url_for
import threading
from face_detection import capture_images, train_recognizer

app = Flask(__name__)

# Paths for dataset, model, and labels
DATASET_PATH = "dataset"
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.pickle"

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

@app.route('/capture_faces')
def capture_faces():
    username = request.args.get('username')
    # Start face capturing in a separate thread
    threading.Thread(target=capture_images, args=(username, DATASET_PATH)).start()
    return render_template('capture.html', username=username)

@app.route('/train_model', methods=['POST'])
def train_model():
    username = request.form.get('username')
    # Train the model in the backend
    train_recognizer(DATASET_PATH, MODEL_PATH, LABELS_PATH)
    return f"Face for {username} registered successfully!"

if __name__ == "__main__":
    app.run(debug=True)
