import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define the folder path where your training image dataset is stored
path = 'student_images'

# Directory to save captured images
captured_images_path = 'captured_images'
os.makedirs(captured_images_path, exist_ok=True)

# Track the names of people who have already been recognized
recognized_names = set()

# Function to load images and their class names
def load_images_from_folder(path):
    images = []
    classNames = []
    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, file_name))
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(file_name)[0])
    return images, classNames

# Function to encode all the train images
def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:  # Check if face is detected
            encode_list.append(encodings[0])
    return encode_list

# Load images and their class names
images, classNames = load_images_from_folder(path)

# Encode all train images
encoded_face_train = find_encodings(images)

# Read webcam for real-time recognition
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame for faster processing
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    faces_in_frame = face_recognition.face_locations(img_rgb)
    encoded_faces = face_recognition.face_encodings(img_rgb, faces_in_frame)

    # Compare faces in the frame to the known faces
    for encode_face, face_loc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        face_dist = face_recognition.face_distance(encoded_face_train, encode_face)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = classNames[match_index].capitalize()

            # Check if the name is already recognized
            if name not in recognized_names:
                recognized_names.add(name)  # Add the name to the set to avoid repeated captures

                # Save the captured image with the name of the person
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Draw a rectangle around the face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Save the image only once per person
                capture_filename = os.path.join(captured_images_path, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(capture_filename, img)
                print(f"Image saved: {capture_filename}")

    # Display the resulting frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()