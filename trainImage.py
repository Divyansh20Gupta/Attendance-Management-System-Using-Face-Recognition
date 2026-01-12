import os
import cv2
import numpy as np
from PIL import Image

def TrainImage(haar_path, trainimage_path, train_label_path, message, text_to_speech):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(haar_path)

        faces, ids = getImagesAndLabels(trainimage_path, detector)

        if len(faces) == 0:
            msg = "No face images found. Please register a student first."
            message.configure(text=msg, fg="yellow")
            text_to_speech(msg)
            return

        recognizer.train(faces, np.array(ids))
        recognizer.save(train_label_path)

        msg = "Training Completed Successfully."
        message.configure(text=msg, fg="yellow")
        text_to_speech(msg)

    except Exception as e:
        print("Training Error:", e)
        msg = "Training failed. Check console."
        message.configure(text=msg, fg="red")
        text_to_speech(msg)


def getImagesAndLabels(root_path, detector):
    face_samples = []
    ids = []

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)

        if not os.path.isdir(folder_path):
            continue

        try:
            enrollment = int(folder.split("_")[0])
        except:
            print("Skipping invalid folder:", folder)
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)

            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            try:
                img = Image.open(img_path).convert("L")
                gray = np.array(img, "uint8")

                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))

                    face_samples.append(face_roi)
                    ids.append(enrollment)

            except Exception as e:
                print("Error reading:", img_path, e)

    return face_samples, ids
