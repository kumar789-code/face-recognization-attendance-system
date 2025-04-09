import cv2
import numpy as np
import os

def train_model():
    data_path = 'dataset'
    face_data = []
    labels = []
    label_map = {}

    if not os.path.exists(data_path):
        print("Dataset folder not found.")
        return

    people = sorted(os.listdir(data_path))  # Sorted for consistent label order
    for label, person in enumerate(people):
        label_map[label] = person
        person_path = os.path.join(data_path, person)

        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipped unreadable image: {img_path}")
                continue

            img = cv2.resize(img, (200, 200))  # Ensure uniform size
            face_data.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)

    if not face_data:
        print("No face data found. Please capture images first.")
        return

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_data, np.array(labels))
    face_recognizer.save('trained_model.yml')

    with open("labels.txt", "w") as f:
        for label, name in label_map.items():
            f.write(f"{label},{name}\n")

    print(f"Model trained on {len(set(labels))} people with {len(face_data)} images.")
    print("Model and labels saved successfully.")

train_model()
