import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time


def TakeImage(enroll, name, haar_path, img_dir, message, err_screen, text_to_speech):

    # --------- VALIDATION ----------
    if enroll == "" and name == "":
        t = "Please enter Enrollment Number and Name."
        text_to_speech(t)
        message.configure(text=t, fg="yellow")
        return

    if enroll == "":
        t = "Please enter Enrollment Number."
        text_to_speech(t)
        message.configure(text=t, fg="yellow")
        return

    if name == "":
        t = "Please enter Name."
        text_to_speech(t)
        message.configure(text=t, fg="yellow")
        return

    # --------- CREATE DIRECTORY FOR IMAGES ----------
    folder_name = f"{enroll}_{name}"
    save_path = os.path.join(img_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # --------- START CAMERA ----------
    try:
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(haar_path)
        sampleNum = 0

        if not cam.isOpened():
            msg = "Camera not detected."
            text_to_speech(msg)
            message.config(text=msg, fg="yellow")
            return

        while True:
            ret, img = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1

                # save face crop
                img_path = os.path.join(
                    save_path,
                    f"{name}_{enroll}_{sampleNum}.jpg"
                )

                face_crop = gray[y:y+h, x:x+w]
                cv2.imwrite(img_path, face_crop)

                # rectangle on screen
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow("Capturing Images (press 'q' to stop)", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
        text_to_speech("Error capturing images.")
        return

    # --------- SAVE STUDENT DETAILS ----------
    try:
        csv_file = "StudentDetails/studentdetails.csv"

        # Create file with header if not exists
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Enrollment", "Name"])

        # Append student
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([enroll, name])

    except Exception as e:
        print("CSV error:", e)
        text_to_speech("Error saving student details.")
        return

    # --------- SUCCESS MESSAGE ----------
    msg = f"Images saved for {name} ({enroll})."
    message.configure(text=msg, fg="yellow")
    text_to_speech(msg)
