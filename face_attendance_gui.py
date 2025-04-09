import cv2
import numpy as np
import os
import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog

# Load face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ========== 1. Capture Faces ==========
def capture_faces():
    name = simpledialog.askstring("Input", "Enter the person's name:")
    if not name:
        return

    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, name)

    if not os.path.exists(person_path):
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Captured", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) == ord('q') or count >=100:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Face data for {name} captured successfully!")

# ========== 2. Train Model ==========
def train_model():
    data_path = 'dataset'
    face_data, labels = [], []
    label_map = {}

    people = os.listdir(data_path)
    for label, person in enumerate(people):
        label_map[label] = person
        person_path = os.path.join(data_path, person)
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            face_data.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(face_data, np.array(labels))
    model.save('trained_model.yml')

    with open("labels.txt", "w") as f:
        for label, name in label_map.items():
            f.write(f"{label},{name}\n")

    messagebox.showinfo("Info", "Model trained and saved!")

# ========== 3. Face Attendance ==========
def mark_attendance(name):
    now = datetime.datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = "attendance.csv"

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Name,Time\n")

    with open(filename, "r+") as f:
        data = f.read()
        if name not in data:
            f.write(f"{name},{time_string}\n")

def start_attendance():
    if not os.path.exists("trained_model.yml"):
        messagebox.showerror("Error", "Please train the model first!")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("trained_model.yml")

    label_map = {}
    with open("labels.txt", "r") as f:
        for line in f:
            label, name = line.strip().split(",")
            label_map[int(label)] = name

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            label, confidence = model.predict(roi)

            if confidence > 30:
                name = label_map[label]
                mark_attendance(name)
                text = f"{name} ({round(confidence,2)})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== GUI ==========
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("400x300")

tk.Label(root, text="Face Attendance System", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="Capture Face", width=20, command=capture_faces).pack(pady=10)
tk.Button(root, text="Train Model", width=20, command=train_model).pack(pady=10)
tk.Button(root, text="Start Attendance", width=20, command=start_attendance).pack(pady=10)
tk.Button(root, text="Exit", width=20, command=root.quit).pack(pady=10)

root.mainloop()
