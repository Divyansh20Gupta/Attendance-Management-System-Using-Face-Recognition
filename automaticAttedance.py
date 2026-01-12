import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import datetime
import pandas as pd
import cv2


# --- PATHS ---
HAAR_PATH = "haarcascade_frontalface_default.xml"
TRAIN_LABEL_PATH = "./TrainingImageLabel/Trainner.yml"
STUDENT_CSV = "./StudentDetails/studentdetails.csv"
ATTENDANCE_DIR = "./Attendance"

os.makedirs(ATTENDANCE_DIR, exist_ok=True)


def subjectChoose(text_to_speech):

    BG = "#0f1113"
    ACCENT = "#f2d300"
    BTN_BG = "#111214"

    win = tk.Toplevel()
    win.title("Attendance (LBPH)")
    win.configure(bg=BG)
    win.geometry("600x380")
    win.resizable(False, False)

    tk.Label(win, text="Enter Subject Name", bg=BG, fg=ACCENT,
             font=("Segoe UI", 20, "bold")).pack(pady=20)

    entry = tk.Entry(win, bg=BTN_BG, fg=ACCENT, insertbackground=ACCENT,
                     font=("Segoe UI", 14, "bold"), relief="flat")
    entry.pack(padx=20, pady=10, ipadx=10, ipady=6)

    notify = tk.Label(win, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 11))
    notify.pack()

    state = {"running": False}

    def safe_tts(msg):
        try:
            text_to_speech(msg)
        except:
            pass

    # ----------------------------------------------------------
    #                     ATTENDANCE THREAD
    # ----------------------------------------------------------
    def attendance_thread(subject):

        state["running"] = True
        notify.config(text="Starting camera...")
        safe_tts(f"Starting attendance for {subject}")

        # Load LBPH model
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAIN_LABEL_PATH)
        except:
            notify.config(text="ERROR: Train the model first.")
            state["running"] = False
            return

        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)

        # Load student names
        try:
            df = pd.read_csv(STUDENT_CSV, dtype={"Enrollment": str})
        except:
            notify.config(text="studentdetails.csv missing")
            state["running"] = False
            return

        label_map = {
            str(r["Enrollment"]): r["Name"] for _, r in df.iterrows()
        }

        # Open webcam
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            notify.config(text="Camera error")
            return

        THRESHOLD = 50
        REQUIRED_TIME = 3.0    # 3 seconds

        found = {}             # Final attendance list
        timers = {}            # {enrollment: start_time}

        notify.config(text="Camera running... Press ESC to exit.")

        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))

                try:
                    pred_label, conf = recognizer.predict(roi)
                except:
                    continue

                enroll = str(pred_label)

                # Unknown logic
                if conf >= THRESHOLD or enroll not in label_map:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue

                name = label_map[enroll]

                # Timer start
                if enroll not in timers:
                    timers[enroll] = time.time()

                elapsed = time.time() - timers[enroll]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({elapsed:.1f}s)", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Mark attendance after 3 seconds
                if elapsed >= REQUIRED_TIME:
                    if enroll not in found:
                        found[enroll] = name
                        safe_tts(f"Attendance marked for {name}")
                        print(f"Marked: {name}")

            cv2.imshow("Attendance - Press ESC", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save attendance if found
        if not found:
            notify.config(text="No attendance recorded.")
            safe_tts("No attendance recorded.")
            state["running"] = False
            return

        date = datetime.datetime.now().strftime("%Y-%m-%d")
        timepart = datetime.datetime.now().strftime("%H-%M-%S")

        rows = [
            {"Enrollment": e, "Name": n, "Date": date, "Present": 1}
            for e, n in found.items()
        ]

        df_att = pd.DataFrame(rows)

        # Subject folder
        folder = os.path.join(ATTENDANCE_DIR, subject)
        os.makedirs(folder, exist_ok=True)

        filepath = os.path.join(folder, f"{subject}_{date}_{timepart}.csv")
        df_att.to_csv(filepath, index=False)

        notify.config(text=f"Attendance saved: {os.path.basename(filepath)}")
        safe_tts("Attendance saved successfully!")

        show_csv(filepath)
        state["running"] = False

    # ----------------------------------------------------------
    #                    SHOW CSV WINDOW (FIXED)
    # ----------------------------------------------------------
    def show_csv(path):
        df = pd.read_csv(path)

        tv = tk.Toplevel(win)
        tv.title("Attendance Sheet")
        tv.geometry("600x400")

        # Scrollbars
        frame = tk.Frame(tv)
        frame.pack(fill="both", expand=True)

        tree_scroll_y = tk.Scrollbar(frame)
        tree_scroll_y.pack(side="right", fill="y")

        tree_scroll_x = tk.Scrollbar(frame, orient="horizontal")
        tree_scroll_x.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            frame,
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        tree.pack(fill="both", expand=True)

        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)

        # Define columns
        tree["columns"] = df.columns.tolist()

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor="center")

        # Insert rows (FIXED)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    # ----------------------------------------------------------
    #                       BUTTONS
    # ----------------------------------------------------------
    def start():
        subject = entry.get().strip()
        if not subject:
            notify.config(text="Enter subject name")
            return

        if state["running"]:
            notify.config(text="Already running...")
            return

        threading.Thread(target=attendance_thread,
                         args=(subject,), daemon=True).start()

    tk.Button(
        win, text="Start Attendance", command=start,
        bg=BTN_BG, fg=ACCENT, padx=15, pady=8
    ).pack(pady=10)

    tk.Button(
        win, text="Close", command=win.destroy,
        bg="#222", fg=ACCENT, padx=15, pady=8
    ).pack(pady=10)

    entry.focus_set()
    win.grab_set()
