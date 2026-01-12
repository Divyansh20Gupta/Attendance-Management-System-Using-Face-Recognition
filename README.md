# Attendance-Management-System-Using-Face-Recognition
🎓 A Python-based attendance management system using real-time face recognition with OpenCV and LBPH algorithm. Automates attendance, prevents proxy, and stores records in CSV format.
It eliminates manual attendance and prevents proxy attendance using computer vision techniques.

---

## 🚀 Features
- Real-time face detection and recognition
- Automatic attendance marking
- CSV-based attendance records
- Subject-wise attendance storage
- Simple and user-friendly GUI (Tkinter)
- Uses LBPH algorithm for recognition

---

## 🛠️ Technologies Used
- Python 3.11
- OpenCV
- NumPy
- Pandas
- Tkinter
- Pillow (PIL)
- Haar Cascade Classifier

---

## 📁 Project Modules
| File | Description |
|-----|------------|
| `takeImage.py` | Capture student face images |
| `trainImage.py` | Train LBPH face recognition model |
| `automaticAttedance.py` | Automatic attendance using camera |
| `main.py` | Main GUI launcher |
| `Trainner.yml` | Trained face recognition model |

---

## 🧠 Face Recognition Algorithm
This project uses **LBPH (Local Binary Pattern Histogram)** for face recognition because:
- It works well in real-time
- Handles lighting variations
- Suitable for small datasets

---

## 📸 How It Works
1. Capture images of students using webcam
2. Train the system using LBPH algorithm
3. Detect and recognize faces in real time
4. Mark attendance automatically in CSV file

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
python main.py
