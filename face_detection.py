import cv2
import os
import numpy as np
import pickle

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture images live
def capture_images(person_name, save_path, image_limit=50):
    cap = cv2.VideoCapture(0)
    count = 0  # Counter for captured images

    # Create a folder for the person if it doesn't exist
    person_path = os.path.join(save_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    print(f"Capturing images for {person_name}. Look at the camera...")

    while count < image_limit:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the webcam!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            count += 1
            file_name = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_name, face)  # Save the face image
            print(f"Captured image {count}/{image_limit}")

            # Draw a rectangle around the face and display the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        if count >= image_limit:
            print("Image capture complete!")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train the recognizer
def train_recognizer(dataset_path, model_path, labels_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []  # List to store face images
    labels = []  # List to store corresponding labels
    label_dict = {}  # Dictionary to map label IDs to person names
    label_id = 0  # Unique ID for each person

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[label_id] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Could not read {image_path}. Skipping.")
                continue

            faces.append(image)
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    with open(labels_path, 'wb') as f:
        pickle.dump(label_dict, f)

    print("Training completed! Model and labels saved.")

# Main function
if __name__ == "__main__":
    dataset_path = "dataset"  # Path to save the images
    model_path = "face_model.yml"  # Path to save the trained model
    labels_path = "labels.pickle"  # Path to save the labels

    # Step 1: Capture images for a new person
    person_name = input("Enter the name of the person: ")
    capture_images(person_name, dataset_path)

    # Step 2: Train the recognizer
    train_recognizer(dataset_path, model_path, labels_path)

    print("The system is ready to recognize faces!")

